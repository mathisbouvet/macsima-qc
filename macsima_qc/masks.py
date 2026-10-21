"""
masks.py
--------
Génération des masques d'image à partir de fichiers ROI Fiji (.zip/.roi).

Fonctions principales
---------------------
- load_image(path)
- load_rois_from_zip(zip_path, extract_folder)
- generate_masks(image, roi_data, output_dir)
"""

import os
import random
import zipfile

import numpy as np
import tifffile
from read_roi import read_roi_file
from skimage import io as skio
from skimage.draw import polygon, polygon_perimeter


def load_image(path: str) -> np.ndarray:
    """
    Charge une image TIFF (ou tout format supporté par scikit-image).

    L'image est renvoyée telle quelle (niveaux de gris 8/16 bits, RGB ou
    multicanal). Seules ses dimensions sont utilisées par generate_masks().

    Parameters
    ----------
    path : str
        Chemin vers le fichier image (.tif, .png, etc.)

    Returns
    -------
    np.ndarray
        Image chargée sous forme de tableau NumPy.

    Raises
    ------
    FileNotFoundError
        Si le fichier est introuvable ou illisible.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image introuvable : {path}")
    try:
        return skio.imread(path)
    except Exception as e:
        raise FileNotFoundError(f"Image illisible : {path} ({e})") from e


def _image_hw(image: np.ndarray) -> tuple[int, int]:
    """
    Renvoie (hauteur, largeur) d'une image 2D, RGB(A) (H, W, 3|4)
    ou multicanal/multipage (C, H, W).
    """
    if image.ndim == 2:
        return image.shape
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        return image.shape[:2]
    return image.shape[-2:]


def load_rois_from_zip(zip_path: str, extract_folder: str = "roi_dezip") -> dict:
    """
    Extrait et charge tous les ROIs depuis un fichier .zip Fiji.

    Parameters
    ----------
    zip_path : str
        Chemin vers le fichier RoiSet.zip.
    extract_folder : str
        Dossier de destination pour l'extraction (créé si inexistant).

    Returns
    -------
    dict
        Dictionnaire {roi_name: roi_data} issu de read_roi_file.
    """
    os.makedirs(extract_folder, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    roi_files = [f for f in os.listdir(extract_folder) if f.endswith(".roi")]
    roi_data = {}
    for roi_file in roi_files:
        roi_path = os.path.join(extract_folder, roi_file)
        roi_data.update(read_roi_file(roi_path))

    print(f"✅ {len(roi_data)} ROIs chargés depuis {zip_path}")
    return roi_data


def generate_masks(image: np.ndarray, roi_data: dict, output_dir: str = ".") -> dict:
    """
    Génère 4 types de masques à partir des ROIs et les sauvegarde en .tif.

    Masques générés
    ---------------
    - mask_1.tif : ROIs avec couleurs aléatoires, fond noir
    - mask_2.tif : ROIs avec couleurs RGB cycliques (R/G/B), fond noir
    - mask_3.tif : ROIs en niveaux de gris cumulatifs
    - mask_4.tif : ROIs en niveaux de gris cumulatifs + contours noirs

    Parameters
    ----------
    image : np.ndarray
        Image de référence (seules ses dimensions H × W sont utilisées).
    roi_data : dict
        Dictionnaire de ROIs issu de load_rois_from_zip().
    output_dir : str
        Dossier de sortie pour les fichiers mask_*.tif.

    Returns
    -------
    dict
        Dictionnaire {nom_masque: chemin_fichier}.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    hw = _image_hw(image)

    # --- Masque 1 : couleurs aléatoires ---
    masked = np.zeros((*hw, 3), dtype=np.uint8)
    for roi in roi_data.values():
        x, y = np.array(roi["x"]), np.array(roi["y"])
        rr, cc = polygon(y, x, shape=hw)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        masked[rr, cc] = color
    path = os.path.join(output_dir, "mask_1.tif")
    tifffile.imwrite(path, masked, photometric="rgb")
    paths["mask_1"] = path

    # --- Masque 2 : couleurs RGB cycliques ---
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    masked = np.zeros((*hw, 3), dtype=np.uint8)
    for idx, roi in enumerate(roi_data.values()):
        x, y = np.array(roi["x"]), np.array(roi["y"])
        rr, cc = polygon(y, x, shape=hw)
        masked[rr, cc] = colors[idx % len(colors)]
    path = os.path.join(output_dir, "mask_2.tif")
    tifffile.imwrite(path, masked, photometric="rgb")
    paths["mask_2"] = path

    # --- Masque 3 : niveaux de gris cumulatifs ---
    gray = np.zeros(hw, dtype=np.uint8)
    for roi in roi_data.values():
        x, y = np.array(roi["x"]), np.array(roi["y"])
        rr, cc = polygon(y, x, shape=hw)
        gray[rr, cc] = np.clip(gray[rr, cc] + 50, 0, 255)
    path = os.path.join(output_dir, "mask_3.tif")
    tifffile.imwrite(path, gray, photometric="minisblack")
    paths["mask_3"] = path

    # --- Masque 4 : niveaux de gris + contours ---
    gray = np.zeros(hw, dtype=np.uint8)
    for roi in roi_data.values():
        x, y = np.array(roi["x"]), np.array(roi["y"])
        rr, cc = polygon(y, x, shape=hw)
        gray[rr, cc] = np.clip(gray[rr, cc] + 50, 0, 255)
        rr_p, cc_p = polygon_perimeter(y, x, shape=hw)
        gray[rr_p, cc_p] = 0
    path = os.path.join(output_dir, "mask_4.tif")
    tifffile.imwrite(path, gray, photometric="minisblack")
    paths["mask_4"] = path

    print(f"✅ 4 masques générés dans : {output_dir}")
    return paths
