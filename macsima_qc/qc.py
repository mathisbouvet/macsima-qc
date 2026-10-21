"""
qc.py
-----
Contrôle qualité de la segmentation automatique MACSiQView par rapport
à une segmentation de référence.

Pipeline
--------
1. Chargement de df_ref (référence intégrée par défaut) et df_test
2. Normalisation StandardScaler (fit sur ref, transform sur test)
3. Analyse de sensibilité de l'Isolation Forest (contamination 0.01 → 0.20)
4. Modèle final avec la contamination choisie
5. Test Mann-Whitney U (OK vs KO) + taille d'effet (delta de Cliff) + correction de
   Benjamini-Hochberg → suggestions d'ajustement MACSiQView

Fonctions principales
---------------------
- load_qc_data(test_path, ref_path, features)
- contamination_sensitivity(X_ref_scaled, X_test_scaled, values, output_dir, show)
- run_isolation_forest(X_ref_scaled, X_test_scaled, df_test_clean, contamination, output_dir, ...)
- run_mann_whitney(df_annotated, features, param_map, export_path, output_dir, alpha, min_effect)
- run_qc_pipeline(test_path, ref_path, contamination, output_dir, ...)
"""

import os
from importlib import resources

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .utils import FEATURES_TO_USE, PARAM_MAP


def _finish_figure(fig, path, show):
    """Sauvegarde la figure, l'affiche si demandé, sinon la ferme."""
    fig.savefig(path, format="png")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _bh_adjust(pvalues) -> np.ndarray:
    """q-values de Benjamini-Hochberg (contrôle du taux de fausses découvertes)."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out


def _effect_magnitude(delta: float) -> str:
    """Seuils de Romano et al. (2006) pour |delta de Cliff|."""
    d = abs(delta)
    if d < 0.147:
        return "négligeable"
    if d < 0.33:
        return "faible"
    if d < 0.474:
        return "moyenne"
    return "forte"


BUILTIN_REFERENCE = "reference_segmentation.csv"


def _load_builtin_reference(features: list[str]) -> pd.DataFrame:
    """
    Charge la référence intégrée au package (macsima_qc/data/reference_segmentation.csv).
    """
    ref = resources.files("macsima_qc.data").joinpath(BUILTIN_REFERENCE)
    if not ref.is_file():
        raise FileNotFoundError(
            f"La référence intégrée ({BUILTIN_REFERENCE}) est absente de cette installation "
            "de macsima-qc. Passez une référence explicite via ref_path=..., ou mettez à jour "
            "le package (pip install -U macsima-qc)."
        )
    with ref.open("r") as f:
        df = pd.read_csv(f)

    df.columns = df.columns.str.strip()
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans la référence intégrée : {missing}")

    print(f"📦 Référence intégrée chargée ({len(df)} cellules)")
    return df[features].dropna()


def load_qc_data(
    test_path: str,
    ref_path: str = None,
    features: list[str] = None,
) -> tuple:
    """
    Charge et normalise les données de test et de référence.

    Si ref_path n'est pas fourni, la référence intégrée au package est utilisée.
    Le StandardScaler est fitté uniquement sur X_ref.

    Parameters
    ----------
    test_path : str
        Chemin vers le CSV de segmentation à évaluer.
    ref_path : str, optional
        Chemin vers un CSV de référence personnalisé.
        Si None, la référence intégrée est utilisée.
    features : list[str], optional
        Features morphologiques à utiliser. Par défaut : FEATURES_TO_USE.

    Returns
    -------
    tuple : (X_ref_scaled, X_test_scaled, df_test_clean, scaler, features)
    """
    if features is None:
        features = FEATURES_TO_USE

    # Chargement référence
    if ref_path is None:
        X_ref = _load_builtin_reference(features)
    else:
        df_ref = pd.read_csv(ref_path)
        df_ref.columns = df_ref.columns.str.strip()
        missing = [c for c in features if c not in df_ref.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans la référence : {missing}")
        X_ref = df_ref[features].dropna()
        print(f"📂 Référence personnalisée chargée ({len(X_ref)} cellules)")

    # Chargement test
    df_test = pd.read_csv(test_path)
    df_test.columns = df_test.columns.str.strip()
    missing = [c for c in features if c not in df_test.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier test : {missing}")

    X_test = df_test[features].dropna()
    df_test_clean = df_test.loc[X_test.index].copy()
    n_dropped = len(df_test) - len(X_test)
    if n_dropped:
        print(
            f"⚠️  {n_dropped} cellule(s) test avec des valeurs manquantes exclue(s) de l'analyse "
            "(Segmentation_OK = NaN dans les résultats du pipeline)"
        )

    scaler = StandardScaler()
    X_ref_scaled = scaler.fit_transform(X_ref)
    X_test_scaled = scaler.transform(X_test)

    print(f"✅ Référence : {len(X_ref)} cellules · Test : {len(X_test)} cellules")
    return X_ref_scaled, X_test_scaled, df_test_clean, scaler, features


def contamination_sensitivity(
    X_ref_scaled,
    X_test_scaled,
    values: list[float] = None,
    output_dir: str = "figures",
    show: bool = True,
) -> list[float]:
    """
    Analyse de sensibilité de l'Isolation Forest selon le paramètre de contamination.

    Parameters
    ----------
    X_ref_scaled : array-like
    X_test_scaled : array-like
    values : list[float]
        Valeurs de contamination à tester. Par défaut : [0.01, 0.05, 0.10, 0.15, 0.20].
    output_dir : str
    show : bool
        Afficher la figure (True) ou seulement la sauvegarder (False).

    Returns
    -------
    list[float]
        Pourcentages de cellules "OK" pour chaque valeur de contamination.
    """
    if values is None:
        values = [0.01, 0.05, 0.10, 0.15, 0.20]

    os.makedirs(output_dir, exist_ok=True)
    ok_percentages = []

    for c in values:
        model = IsolationForest(contamination=c, random_state=42)
        model.fit(X_ref_scaled)
        preds = model.predict(X_test_scaled)
        ok_pct = (preds == 1).sum() / len(preds) * 100
        ok_percentages.append(ok_pct)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(values, ok_percentages, marker="o", color="steelblue")
    ax.set_xlabel("Contamination rate")
    ax.set_ylabel("% well-segmented cells")
    ax.set_title("Contamination sensitivity analysis")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    out = os.path.join(output_dir, "contamination_sensitivity.png")
    _finish_figure(fig, out, show)

    print("\nContamination · % OK")
    for c, pct in zip(values, ok_percentages):
        print(f"  {c:.2f} → {pct:.1f}%")
    print(f"💾 Figure sauvegardée : {out}")

    return ok_percentages


def run_isolation_forest(
    X_ref_scaled,
    X_test_scaled,
    df_test_clean: pd.DataFrame,
    contamination: float = 0.10,
    output_dir: str = "figures",
    export_path: str = None,
    df_test_full: pd.DataFrame = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Entraîne l'Isolation Forest final et annote les cellules test.

    Parameters
    ----------
    X_ref_scaled : array-like
    X_test_scaled : array-like
    df_test_clean : pd.DataFrame
    contamination : float
    output_dir : str
        Dossier de sortie (figure + CSV annoté).
    export_path : str, optional
        Chemin du CSV annoté. Par défaut : <output_dir>/segmentation_test_annotated.csv.
    df_test_full : pd.DataFrame, optional
        CSV test complet, y compris les cellules exclues pour valeurs manquantes.
        Si fourni, la sortie a une ligne par cellule du CSV d'origine, dans le même
        ordre ; les cellules exclues ont Segmentation_OK et Anomaly_Score à NaN.
    show : bool
        Afficher la figure (True) ou seulement la sauvegarder (False).

    Returns
    -------
    pd.DataFrame
        df_test_clean (ou df_test_full) enrichi de Segmentation_OK (1 = OK, -1 = KO)
        et Anomaly_Score.
    """
    os.makedirs(output_dir, exist_ok=True)
    if export_path is None:
        export_path = os.path.join(output_dir, "segmentation_test_annotated.csv")

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_ref_scaled)
    preds = model.predict(X_test_scaled)
    scores = model.decision_function(X_test_scaled)

    if df_test_full is None:
        df_out = df_test_clean.copy()
    else:
        df_out = df_test_full.copy()
    df_out["Segmentation_OK"] = pd.Series(preds, index=df_test_clean.index).astype("Int64")
    df_out["Anomaly_Score"] = pd.Series(scores, index=df_test_clean.index)

    nb_total = len(preds)
    nb_valides = (preds == 1).sum()
    pct = nb_valides / nb_total * 100
    print(f"\n✅ Résultat avec contamination={contamination:.2f} :")
    print(f"   {nb_valides} / {nb_total} cellules bien segmentées ({pct:.2f}%)")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(scores, bins=50, color="steelblue", edgecolor="white")
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Decision threshold")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Isolation Forest Anomaly Scores")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(output_dir, "anomaly_scores_distribution.png")
    _finish_figure(fig, out, show)

    df_out.to_csv(export_path, index=False)
    print(f"💾 Résultats sauvegardés : {export_path}")

    return df_out


MW_COLUMNS = [
    "Variable",
    "Médiane OK",
    "Médiane KO",
    "Différence",
    "Delta de Cliff",
    "Taille d'effet",
    "Paramètre MACSiQ lié",
    "Suggestion d'ajustement",
    "Écart KO vs OK (médianes)",
    "p-value",
    "q-value (BH)",
]


def run_mann_whitney(
    df_annotated: pd.DataFrame,
    features: list[str] = None,
    param_map: dict = None,
    export_path: str = None,
    output_dir: str = "figures",
    alpha: float = 0.01,
    min_effect: float = 0.33,
) -> pd.DataFrame:
    """
    Compare les cellules OK vs KO feature par feature et génère des suggestions
    d'ajustement des paramètres MACSiQView.

    Pour chaque feature :
    - test de Mann-Whitney U bilatéral (OK vs KO) ;
    - taille d'effet : delta de Cliff, δ = P(KO > OK) − P(KO < OK), dans [−1, 1]
      (δ > 0 : valeurs plus grandes chez les KO) ;
    - correction de Benjamini-Hochberg sur l'ensemble des features (q-value).

    Avec des milliers de cellules, presque toutes les différences deviennent
    « significatives » : une suggestion n'est donc émise que si la différence est
    à la fois significative (q < alpha) ET d'ampleur suffisante (|δ| ≥ min_effect).

    Parameters
    ----------
    df_annotated : pd.DataFrame
        Sortie de run_isolation_forest (colonne Segmentation_OK : 1 / -1 / NaN).
    features : list[str]
    param_map : dict
    export_path : str, optional
        Chemin du CSV de suggestions. Par défaut : <output_dir>/macsiq_param_suggestions.csv.
    output_dir : str
        Dossier de sortie utilisé si export_path n'est pas fourni.
    alpha : float
        Seuil sur la q-value (Benjamini-Hochberg). Par défaut 0.01.
    min_effect : float
        |δ de Cliff| minimal pour émettre une suggestion. Par défaut 0.33
        (effet « moyen » selon Romano et al., 2006).

    Returns
    -------
    pd.DataFrame
        Une ligne par feature testée, triée par |δ| décroissant.
    """
    if features is None:
        features = FEATURES_TO_USE
    if param_map is None:
        param_map = PARAM_MAP
    if export_path is None:
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, "macsiq_param_suggestions.csv")

    present = [f for f in features if f in df_annotated.columns]
    df = df_annotated.dropna(subset=present + ["Segmentation_OK"])
    is_ok = df["Segmentation_OK"] == 1
    is_ko = df["Segmentation_OK"] == -1

    rows = []
    for feature in present:
        ok_vals = df.loc[is_ok, feature].to_numpy(dtype=float)
        ko_vals = df.loc[is_ko, feature].to_numpy(dtype=float)
        if len(ok_vals) < 10 or len(ko_vals) < 10:
            continue

        u_ko, p = mannwhitneyu(ko_vals, ok_vals, alternative="two-sided")
        delta = 2.0 * u_ko / (len(ko_vals) * len(ok_vals)) - 1.0
        med_ok, med_ko = float(np.median(ok_vals)), float(np.median(ko_vals))
        rows.append({
            "Variable": feature,
            "Médiane OK": round(med_ok, 3),
            "Médiane KO": round(med_ko, 3),
            "Delta de Cliff": round(delta, 3),
            "Taille d'effet": _effect_magnitude(delta),
            "Paramètre MACSiQ lié": param_map.get(feature, ("-", "-", "-"))[0],
            "Écart KO vs OK (médianes)": (
                f"{(med_ko / med_ok - 1) * 100:+.1f} %" if med_ok != 0 else "-"
            ),
            "p-value": p,
            "_suggest": param_map.get(feature, ("-", "-", "-")),
        })

    q = _bh_adjust([r["p-value"] for r in rows])
    for r, qv in zip(rows, q):
        r["q-value (BH)"] = qv
        significant = qv < alpha
        r["Différence"] = "-"
        r["Suggestion d'ajustement"] = "-"
        if significant and r["Delta de Cliff"] != 0:
            r["Différence"] = "KO < OK" if r["Delta de Cliff"] < 0 else "KO > OK"
            if abs(r["Delta de Cliff"]) >= min_effect:
                r["Suggestion d'ajustement"] = r["_suggest"][1 if r["Delta de Cliff"] < 0 else 2]
        del r["_suggest"]

    summary_df = pd.DataFrame(rows, columns=MW_COLUMNS)
    if not summary_df.empty:
        summary_df = (
            summary_df.assign(_abs=summary_df["Delta de Cliff"].abs())
            .sort_values("_abs", ascending=False)
            .drop(columns="_abs")
            .reset_index(drop=True)
        )

    if summary_df.empty:
        n_ok, n_ko = int(is_ok.sum()), int(is_ko.sum())
        print(
            f"\n⚠️  Comparaison OK/KO impossible ({n_ok} OK · {n_ko} KO, il en faut ≥ 10 de chaque). "
            "Si presque tout est KO, la référence est probablement inadaptée à ce tissu : "
            "fournissez la vôtre via ref_path."
        )
    else:
        n_sugg = int((summary_df["Suggestion d'ajustement"] != "-").sum())
        print(
            f"\n🧠 Paramètres MACSiQ à ajuster (q < {alpha}, |δ| ≥ {min_effect}) : "
            f"{n_sugg} suggestion(s)\n"
        )
        fmt = {c: "{:.2g}".format for c in ("p-value", "q-value (BH)")}
        print(summary_df.to_string(index=False, formatters=fmt))

    summary_df.to_csv(export_path, index=False)
    print(f"\n💾 Résumé exporté : {export_path}")

    return summary_df


def run_qc_pipeline(
    test_path: str,
    ref_path: str = None,
    contamination: float = 0.10,
    features: list[str] = None,
    output_dir: str = "figures",
    sensitivity_values: list[float] = None,
    alpha: float = 0.01,
    min_effect: float = 0.33,
    show: bool = True,
) -> tuple:
    """
    Pipeline QC complet : chargement → sensibilité → Isolation Forest → Mann-Whitney.

    Parameters
    ----------
    test_path : str
        CSV de segmentation à évaluer.
    ref_path : str, optional
        CSV de référence personnalisé. Si None, la référence intégrée est utilisée.
    contamination : float
        Taux de contamination pour l'Isolation Forest final.
    features : list[str]
        Features à utiliser (par défaut : FEATURES_TO_USE).
    output_dir : str
        Dossier de sortie pour toutes les figures et les deux CSV
        (segmentation_test_annotated.csv, macsiq_param_suggestions.csv).
    sensitivity_values : list[float]
    alpha, min_effect : float
        Seuils des suggestions MACSiQView (voir run_mann_whitney).
    show : bool
        Afficher les figures (True) ou seulement les sauvegarder (False).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_annotated, df_suggestions). df_annotated a une ligne par cellule du CSV
        test, dans le même ordre ; les cellules avec des valeurs manquantes sur les
        features ne sont pas évaluées (Segmentation_OK = NaN).
    """
    X_ref_s, X_test_s, df_test_clean, _, feats = load_qc_data(test_path, ref_path, features)
    df_test_full = pd.read_csv(test_path)
    df_test_full.columns = df_test_full.columns.str.strip()

    contamination_sensitivity(X_ref_s, X_test_s, sensitivity_values, output_dir, show=show)
    df_annotated = run_isolation_forest(
        X_ref_s, X_test_s, df_test_clean, contamination, output_dir,
        df_test_full=df_test_full, show=show,
    )
    df_suggestions = run_mann_whitney(
        df_annotated, feats, output_dir=output_dir, alpha=alpha, min_effect=min_effect,
    )
    return df_annotated, df_suggestions