"""
utils.py
--------
Constantes partagées et utilitaires généraux du package macsima_qc.
"""

import importlib
import subprocess
import sys

# Colonnes morphologiques de base (segmentation manuelle Fiji)
MANUAL_COLUMNS = ["area", "perimeter", "centroid_x", "centroid_y", "feret", "mean_intensity"]

# Features morphologiques utilisées pour le score KS (distance moyenne).
# centroid_x / centroid_y sont exclus : la position d'une cellule dépend du champ
# imagé, pas de la qualité de la segmentation.
KS_FEATURES = ["area", "perimeter", "feret", "mean_intensity"]

# Mapping colonnes Fiji → noms normalisés
MANUAL_RENAME_MAP = {
    "Area": "area",
    "Perim.": "perimeter",
    "XM": "centroid_x",
    "YM": "centroid_y",
    "Feret": "feret",
    "Mean": "mean_intensity",
}

# Mapping colonnes MACSiQView → noms normalisés
AUTO_RENAME_MAP = {
    "Nucleus Size": "area",
    "Nucleus Contour Length": "perimeter",
    "Nuc X": "centroid_x",
    "Nuc Y": "centroid_y",
    "Nucleus Feret Diameter Max": "feret",
    "Nucleus DNA Mean": "mean_intensity",
}

# 14 features morphologiques pour l'Isolation Forest
FEATURES_TO_USE = [
    "Cell Bbox X Size",
    "Cell Bbox Y Size",
    "Cell Shape Circle Like",
    "Cell Shape Ellipse Like",
    "Cell Shape Elongation",
    "Cell Shape Square Like",
    "Cell Shape Triangle Like",
    "Cell Size",
    "Nucleus Size",
    "Nucleus Roundness",
    "Nucleus Convexity",
    "Cell Convexity",
    "Quality Cell In-Focus",
    "Quality Nuclear Segmentation",
]

# Dictionnaire param_map : feature → (paramètre MACSiQ, suggestion si KO<OK, suggestion si KO>OK)
PARAM_MAP = {
    "Nucleus Size": ("Diamètre min / max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Size": ("Diamètre min / max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Bbox X Size": ("Diamètre max", "↑ diamètre min", "↑ diamètre max"),
    "Cell Bbox Y Size": ("Diamètre max", "↑ diamètre min", "↑ diamètre max"),
    "Nucleus Roundness": ("Séparation / Smoothing", "↑ séparation", "↓ séparation ou ↓ smoothing"),
    "Nucleus Convexity": ("Smoothing filter sigma", "↑ sigma", "↓ sigma"),
    "Cell Convexity": ("Smoothing filter sigma", "↑ sigma", "↓ sigma"),
    "Cell Shape Ellipse Like": ("Contours / Smoothing", "↑ smoothing", "↓ smoothing"),
    "Cell Shape Circle Like": ("Contours / Smoothing", "↑ smoothing", "↓ smoothing"),
    "Cell Shape Elongation": ("Contours / Séparation", "↓ séparation", "↑ séparation"),
    "Cell Shape Square Like": ("Contours", "-", "-"),
    "Cell Shape Triangle Like": ("Contours", "-", "-"),
    "Quality Cell In-Focus": ("Qualité image (acquisition)", "-", "-"),
    "Quality Nuclear Segmentation": ("Sensibilité / Smoothing", "↑ sensibilité", "↓ sensibilité"),
}


def check_and_install_dependencies():
    """
    Vérifie et installe automatiquement les dépendances requises.
    Utile principalement en environnement Colab.
    """
    required = {
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "skimage": "scikit-image",
        "read_roi": "read-roi",
        "pandas": "pandas",
        "sklearn": "scikit-learn",
        "scipy": "scipy",
        "seaborn": "seaborn",
        "tifffile": "tifffile",
    }
    print("🧪 Vérification de l'environnement...")
    for module_name, package_name in required.items():
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {module_name} OK")
        except ImportError:
            print(f"  ⚠️  {module_name} manquant — installation de {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"  🚀 {package_name} installé.")
    print("✨ Environnement prêt.")