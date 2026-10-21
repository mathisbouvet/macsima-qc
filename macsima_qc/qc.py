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
5. Test Mann-Whitney U + génération de suggestions d'ajustement MACSiQView

Fonctions principales
---------------------
- load_qc_data(test_path, ref_path, features)
- contamination_sensitivity(X_ref_scaled, X_test_scaled, values, output_dir)
- run_isolation_forest(X_ref_scaled, X_test_scaled, df_test_clean, contamination, output_dir)
- run_mann_whitney(df_annotated, features, param_map, export_path, output_dir)
- run_qc_pipeline(test_path, ref_path, contamination, output_dir)
"""

import os
import warnings
from importlib import resources

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .utils import FEATURES_TO_USE, PARAM_MAP


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

    warnings.filterwarnings("ignore")

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

    plt.figure(figsize=(8, 4))
    plt.plot(values, ok_percentages, marker="o", color="steelblue")
    plt.xlabel("Contamination rate")
    plt.ylabel("% well-segmented cells")
    plt.title("Contamination sensitivity analysis")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out = os.path.join(output_dir, "contamination_sensitivity.png")
    plt.savefig(out, format="png")
    plt.show()

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

    Returns
    -------
    pd.DataFrame
        df_test_clean enrichi de Segmentation_OK et Anomaly_Score.
    """
    os.makedirs(output_dir, exist_ok=True)
    if export_path is None:
        export_path = os.path.join(output_dir, "segmentation_test_annotated.csv")

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X_ref_scaled)
    preds = model.predict(X_test_scaled)
    scores = model.decision_function(X_test_scaled)

    df_out = df_test_clean.copy()
    df_out["Segmentation_OK"] = preds
    df_out["Anomaly_Score"] = scores

    nb_total = len(preds)
    nb_valides = (preds == 1).sum()
    pct = nb_valides / nb_total * 100
    print(f"\n✅ Résultat avec contamination={contamination:.2f} :")
    print(f"   {nb_valides} / {nb_total} cellules bien segmentées ({pct:.2f}%)")

    plt.figure(figsize=(10, 4))
    plt.hist(scores, bins=50, color="steelblue", edgecolor="white")
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Decision threshold")
    plt.xlabel("Anomaly Score")
    plt.ylabel("Count")
    plt.title("Distribution of Isolation Forest Anomaly Scores")
    plt.legend()
    plt.tight_layout()

    out = os.path.join(output_dir, "anomaly_scores_distribution.png")
    plt.savefig(out, format="png")
    plt.show()

    df_out.to_csv(export_path, index=False)
    print(f"💾 Résultats sauvegardés : {export_path}")

    return df_out


def run_mann_whitney(
    df_annotated: pd.DataFrame,
    features: list[str] = None,
    param_map: dict = None,
    export_path: str = None,
    output_dir: str = "figures",
) -> pd.DataFrame:
    """
    Compare les cellules OK vs KO par test Mann-Whitney U et génère des
    suggestions d'ajustement des paramètres MACSiQView.

    Parameters
    ----------
    df_annotated : pd.DataFrame
    features : list[str]
    param_map : dict
    export_path : str, optional
        Chemin du CSV de suggestions. Par défaut : <output_dir>/macsiq_param_suggestions.csv.
    output_dir : str
        Dossier de sortie utilisé si export_path n'est pas fourni.

    Returns
    -------
    pd.DataFrame
    """
    if features is None:
        features = FEATURES_TO_USE
    if param_map is None:
        param_map = PARAM_MAP
    if export_path is None:
        os.makedirs(output_dir, exist_ok=True)
        export_path = os.path.join(output_dir, "macsiq_param_suggestions.csv")

    df = df_annotated.copy()
    df = df.dropna(subset=features + ["Segmentation_OK"])
    df["Segmentation_Label"] = df["Segmentation_OK"].map({1: "OK", -1: "KO"})

    summary = []
    for feature in features:
        if feature not in df.columns:
            continue

        ok_vals = df[df["Segmentation_Label"] == "OK"][feature]
        ko_vals = df[df["Segmentation_Label"] == "KO"][feature]

        if len(ok_vals) < 10 or len(ko_vals) < 10:
            continue

        stat, p = mannwhitneyu(ok_vals, ko_vals, alternative="two-sided")
        mean_ok = ok_vals.mean()
        mean_ko = ko_vals.mean()

        direction = "-"
        suggestion = "-"
        percent_change = "-"
        param = param_map.get(feature, ("-", "-", "-"))[0]

        if p < 0.01:
            if mean_ko < mean_ok:
                direction = "KO < OK"
                suggestion = param_map.get(feature, ("-", "-", "-"))[1]
                percent_change = f"+{round((1 - mean_ko / mean_ok) * 100, 1)} %"
            elif mean_ko > mean_ok:
                direction = "KO > OK"
                suggestion = param_map.get(feature, ("-", "-", "-"))[2]
                percent_change = f"+{round((mean_ko / mean_ok - 1) * 100, 1)} %"

        summary.append({
            "Variable": feature,
            "Moyenne OK": round(mean_ok, 2),
            "Moyenne KO": round(mean_ko, 2),
            "Différence": direction,
            "Paramètre MACSiQ lié": param,
            "Suggestion d'ajustement": suggestion,
            "% de changement indicatif": percent_change,
            "p-value": round(p, 4),
        })

    columns = [
        "Variable", "Moyenne OK", "Moyenne KO", "Différence", "Paramètre MACSiQ lié",
        "Suggestion d'ajustement", "% de changement indicatif", "p-value",
    ]
    summary_df = pd.DataFrame(summary, columns=columns).sort_values("p-value")

    if summary_df.empty:
        n_ok = int((df["Segmentation_Label"] == "OK").sum())
        n_ko = int((df["Segmentation_Label"] == "KO").sum())
        print(
            f"\n⚠️  Comparaison OK/KO impossible ({n_ok} OK · {n_ko} KO, il en faut ≥ 10 de chaque). "
            "Si presque tout est KO, la référence est probablement inadaptée à ce tissu : "
            "fournissez la vôtre via ref_path."
        )
    else:
        print("\n🧠 Résumé des paramètres MACSiQ à ajuster :\n")
        print(summary_df.to_string(index=False))

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

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_annotated, df_suggestions)
    """
    X_ref_s, X_test_s, df_test_clean, _, feats = load_qc_data(test_path, ref_path, features)
    contamination_sensitivity(X_ref_s, X_test_s, sensitivity_values, output_dir)
    df_annotated = run_isolation_forest(X_ref_s, X_test_s, df_test_clean, contamination, output_dir)
    df_suggestions = run_mann_whitney(df_annotated, feats, output_dir=output_dir)
    return df_annotated, df_suggestions