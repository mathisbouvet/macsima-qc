"""
comparison.py
-------------
Comparaison de segmentations automatiques (MACSiQView) vs segmentation manuelle (Fiji).

Utilise le test de Kolmogorov-Smirnov pour mesurer la distance entre distributions
de features morphologiques, et produit des visualisations (barplot KS, KDE).

Fonctions principales
---------------------
- load_manual(path)
- load_autos(paths)
- normalize_datasets(df_manual, df_autos)
- compute_ks_distances(df_manual, df_autos, legend, features)
- plot_ks_barplot(distances_df, legend, output_dir, show)
- plot_kde(df_manual, df_autos, legend, output_dir, params, show)
- run_comparison(manual_path, auto_paths, auto_names, output_dir, features, show)
"""

import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler

from .utils import AUTO_RENAME_MAP, KS_FEATURES, MANUAL_COLUMNS, MANUAL_RENAME_MAP


def _finish_figure(fig, path, show, **savefig_kwargs):
    """Sauvegarde la figure, l'affiche si demandé, sinon la ferme (pas de fuite mémoire)."""
    fig.savefig(path, format="png", **savefig_kwargs)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"💾 Figure sauvegardée : {path}")


def load_manual(path: str) -> pd.DataFrame:
    """
    Charge et renomme la segmentation manuelle Fiji.

    Parameters
    ----------
    path : str
        Chemin vers le CSV de segmentation manuelle.

    Returns
    -------
    pd.DataFrame
        DataFrame avec les colonnes normalisées (MANUAL_COLUMNS).
    """
    df = pd.read_csv(path)
    df = df.rename(columns=MANUAL_RENAME_MAP)
    missing = [c for c in MANUAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier manuel : {missing}")
    return df[MANUAL_COLUMNS].copy()


def load_autos(paths: list[str]) -> list[pd.DataFrame]:
    """
    Charge et renomme plusieurs segmentations automatiques MACSiQView.

    Parameters
    ----------
    paths : list[str]
        Liste de chemins vers les CSV de segmentation automatique.

    Returns
    -------
    list[pd.DataFrame]
        Liste de DataFrames avec les colonnes normalisées (MANUAL_COLUMNS).
    """
    dfs = []
    for path in paths:
        df = pd.read_csv(path)
        df = df.rename(columns=AUTO_RENAME_MAP)
        missing = [c for c in MANUAL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans {path} : {missing}")
        dfs.append(df[MANUAL_COLUMNS].copy())
    return dfs


def normalize_datasets(
    df_manual: pd.DataFrame, df_autos: list[pd.DataFrame]
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Normalise tous les datasets via MinMaxScaler, fitté uniquement sur df_manual.

    La référence manuelle est donc ramenée à [0, 1] et les segmentations auto
    sont exprimées sur la même échelle : un décalage de taille entre auto et
    manuel reste visible (valeurs < 0 ou > 1).

    Parameters
    ----------
    df_manual : pd.DataFrame
    df_autos : list[pd.DataFrame]

    Returns
    -------
    tuple[pd.DataFrame, list[pd.DataFrame]]
        (df_manual_scaled, df_autos_scaled)
    """
    scaler = MinMaxScaler()
    df_manual = df_manual.copy()
    df_manual[MANUAL_COLUMNS] = scaler.fit_transform(df_manual[MANUAL_COLUMNS])

    df_autos_scaled = []
    for df in df_autos:
        df = df.copy()
        df[MANUAL_COLUMNS] = scaler.transform(df[MANUAL_COLUMNS])
        df_autos_scaled.append(df)

    return df_manual, df_autos_scaled


def compute_ks_distances(
    df_manual: pd.DataFrame,
    df_autos: list[pd.DataFrame],
    legend: dict,
    features: list[str] = None,
) -> pd.DataFrame:
    """
    Calcule les distances KS entre la segmentation manuelle et chaque segmentation auto.

    La statistique KS ne dépend que des rangs : le résultat est identique sur les
    données brutes ou normalisées (même transformation appliquée aux deux échantillons).

    Parameters
    ----------
    df_manual : pd.DataFrame
    df_autos : list[pd.DataFrame]
    legend : dict
        Mapping {AutoN: nom_lisible}, ex: {"Auto1": "Mask 3 - Single Cell"}
    features : list[str], optional
        Features entrant dans 'Mean Distance'. Par défaut : KS_FEATURES
        (area, perimeter, feret, mean_intensity — sans les centroïdes).

    Returns
    -------
    pd.DataFrame
        Une ligne par segmentation auto, une colonne par feature + 'Mean Distance'.
    """
    if features is None:
        features = KS_FEATURES

    distances = {
        col: [ks_2samp(df_manual[col].dropna(), df_auto[col].dropna())[0] for df_auto in df_autos]
        for col in features
    }
    distances_df = pd.DataFrame(distances, index=list(legend.keys()))
    distances_df["Mean Distance"] = distances_df[features].mean(axis=1)

    best = distances_df["Mean Distance"].idxmin()
    print(f"✅ Segmentation la plus proche de la référence : {legend[best]}")
    return distances_df


def plot_ks_barplot(
    distances_df: pd.DataFrame,
    legend: dict,
    output_dir: str = "figures",
    show: bool = True,
):
    """
    Génère et sauvegarde le barplot des distances KS moyennes (dark theme).

    Le thème sombre est appliqué uniquement à cette figure (le style global
    de matplotlib n'est pas modifié).

    Parameters
    ----------
    distances_df : pd.DataFrame
        Résultat de compute_ks_distances().
    legend : dict
        Mapping {AutoN: nom_lisible}.
    output_dir : str
        Dossier de sortie pour la figure.
    show : bool
        Afficher la figure (True) ou seulement la sauvegarder (False).

    Returns
    -------
    matplotlib.figure.Figure
    """
    os.makedirs(output_dir, exist_ok=True)
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 6))

        distances_df["Mean Distance"].plot(kind="bar", color="#E9EDC9", edgecolor="white", ax=ax)

        for p in ax.patches:
            ax.annotate(
                f"{p.get_height():.3f}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center", va="center", xytext=(0, 10),
                textcoords="offset points", color="white",
            )

        ax.set_title("Average Kolmogorov-Smirnov Distances by Segmentation", color="white")
        ax.set_ylabel("Average Distance", color="white")
        ax.set_xlabel("Segmentations", color="white")
        ax.set_xticks(range(len(distances_df)))
        ax.set_xticklabels(
            [legend[k] for k in distances_df.index], rotation=45, ha="right", color="white"
        )
        ax.tick_params(colors="white")
        ax.grid(True, linestyle="--", alpha=0.6, color="white")
        ax.set_facecolor("black")
        fig.set_facecolor("black")
        fig.tight_layout()

        out = os.path.join(output_dir, "ks_average_distances.png")
        _finish_figure(fig, out, show, facecolor="black")
    return fig


def plot_kde(
    df_manual: pd.DataFrame,
    df_autos: list[pd.DataFrame],
    legend: dict,
    output_dir: str = "figures",
    params: list[str] = None,
    show: bool = True,
):
    """
    Génère et sauvegarde les courbes KDE comparatives.

    Les données sont tracées telles quelles : passez-les par normalize_datasets()
    au préalable (comme le fait run_comparison) pour que toutes les courbes soient
    sur l'échelle de la référence manuelle. Aucune re-normalisation par jeu de
    données n'est faite ici : elle masquerait les décalages de taille entre
    segmentation auto et manuelle.

    Parameters
    ----------
    df_manual : pd.DataFrame
    df_autos : list[pd.DataFrame]
    legend : dict
    output_dir : str
    params : list[str]
        Features à afficher. Par défaut : KS_FEATURES.
    show : bool
        Afficher la figure (True) ou seulement la sauvegarder (False).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if params is None:
        params = KS_FEATURES

    os.makedirs(output_dir, exist_ok=True)

    ncols = 2 if len(params) > 1 else 1
    nrows = math.ceil(len(params) / ncols)
    with plt.style.context("default"):
        fig, axes = plt.subplots(nrows, ncols, figsize=(9 * ncols, 5 * nrows), squeeze=False)
        fig.patch.set_facecolor("white")
        axes = axes.flatten()

        for idx, param in enumerate(params):
            ax = axes[idx]
            ax.set_facecolor("white")
            for i, label in enumerate(legend.values()):
                sns.kdeplot(df_autos[i][param].dropna(), label=label, linewidth=2, ax=ax)
            sns.kdeplot(
                df_manual[param].dropna(),
                label="Manual reference",
                color="cyan",
                linestyle="--",
                linewidth=2,
                ax=ax,
            )
            ax.set_title(f"Comparative Distribution - {param}")
            ax.set_xlabel("Value (scaled to manual reference: 0–1 = manual min–max)")
            ax.set_ylabel("Density")
            ax.grid(True, linestyle="--", alpha=0.6)

        for ax in axes[len(params):]:
            ax.set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Segmentation", loc="upper center", ncol=3)
        fig.tight_layout(rect=[0, 0, 1, 0.93])

        out = os.path.join(output_dir, "kde_comparative_distributions.png")
        _finish_figure(fig, out, show, facecolor="white")
    return fig


def run_comparison(
    manual_path: str,
    auto_paths: list[str],
    auto_names: list[str] = None,
    output_dir: str = "figures",
    features: list[str] = None,
    show: bool = True,
) -> pd.DataFrame:
    """
    Pipeline complet de comparaison : charge, calcule KS, normalise, génère les figures.

    Parameters
    ----------
    manual_path : str
        Chemin vers le CSV de segmentation manuelle.
    auto_paths : list[str]
        Liste de chemins vers les CSV de segmentations automatiques.
    auto_names : list[str]
        Noms lisibles pour chaque segmentation auto (même ordre que auto_paths).
        Par défaut : ["Auto1", "Auto2", ...].
    output_dir : str
        Dossier de sortie pour les figures.
    features : list[str], optional
        Features du score KS et des KDE. Par défaut : KS_FEATURES.
    show : bool
        Afficher les figures (True) ou seulement les sauvegarder (False).

    Returns
    -------
    pd.DataFrame
        Tableau des distances KS.
    """
    if features is None:
        features = KS_FEATURES
    if auto_names is None:
        auto_names = [f"Auto{i+1}" for i in range(len(auto_paths))]
    if len(auto_names) != len(auto_paths):
        raise ValueError("auto_names et auto_paths doivent avoir la même longueur.")

    legend = {f"Auto{i+1}": name for i, name in enumerate(auto_names)}

    df_manual = load_manual(manual_path)
    df_autos = load_autos(auto_paths)

    distances_df = compute_ks_distances(df_manual, df_autos, legend, features)
    distances_df.to_csv(os.path.join(_ensure_dir(output_dir), "ks_distances.csv"))

    df_manual_n, df_autos_n = normalize_datasets(df_manual, df_autos)
    plot_ks_barplot(distances_df, legend, output_dir, show=show)
    plot_kde(df_manual_n, df_autos_n, legend, output_dir, params=features, show=show)

    return distances_df


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path