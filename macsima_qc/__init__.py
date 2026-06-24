"""
macsima_qc
----------
Package Python pour le contrôle qualité de la segmentation cellulaire
sur la plateforme MACSima / MACSiQView.

Modules
-------
- masks       : génération de masques ROI depuis Fiji
- comparison  : comparaison de segmentations via test KS + KDE
- qc          : QC par Isolation Forest + Mann-Whitney U
- utils       : constantes et utilitaires partagés

Usage rapide
------------
# Comparaison de segmentations
from macsima_qc.comparison import run_comparison
distances = run_comparison(
    manual_path="Segmentation_manuelle.csv",
    auto_paths=["Mask_3_Single_Cell.csv", "Mask_4_Import_mask.csv"],
    auto_names=["Mask 3 - Single Cell", "Mask 4 - Import Mask"],
)

# QC Isolation Forest
from macsima_qc.qc import run_qc_pipeline
df_annotated, df_suggestions = run_qc_pipeline(
    ref_path="Segmentation_ref.csv",
    test_path="Segmentation_EH3524.csv",
    contamination=0.10,
)
"""

__version__ = "0.1.4"
__author__ = "Mathis Bouvet"

from .masks import load_image, load_rois_from_zip, generate_masks
from .comparison import run_comparison, compute_ks_distances, plot_ks_barplot, plot_kde
from .qc import run_qc_pipeline, load_qc_data, contamination_sensitivity, run_isolation_forest, run_mann_whitney
from .utils import check_and_install_dependencies, FEATURES_TO_USE, PARAM_MAP

__all__ = [
    # masks
    "load_image",
    "load_rois_from_zip",
    "generate_masks",
    # comparison
    "run_comparison",
    "compute_ks_distances",
    "plot_ks_barplot",
    "plot_kde",
    # qc
    "run_qc_pipeline",
    "load_qc_data",
    "contamination_sensitivity",
    "run_isolation_forest",
    "run_mann_whitney",
    # utils
    "check_and_install_dependencies",
    "FEATURES_TO_USE",
    "PARAM_MAP",
]
