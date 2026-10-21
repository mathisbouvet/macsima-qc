"""
masks.py
--------
Génération des masques d'image à partir de fichiers ROI Fiji (.zip/.roi).

Fonctions principales
---------------------
- load_image(path)
- load_rois_from_zip(zip_path)
- generate_masks(image, roi_data, output_dir)

Types de ROIs Fiji pris en charge : polygon, freehand, traced, rectangle, oval.
"""

import os
import random
import warnings

import numpy as np
import tifffile
from read_roi import read_roi_zip
from skimage import io as skio
from skimage.draw import ellipse, ellipse_perimeter, polygon, polygon_perimeter

POLYGON_TYPES = {"polygon", "freehand", "traced"}
BOX_TYPES = {"rectangle", "oval"}


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


def load_rois_from_zip(zip_path: str, extract_folder: str = None) -> dict:
    """
    Charge tous les ROIs d'un fichier RoiSet.zip Fiji, dans l'ordre du ROI Manager.

    Le zip est lu directement, sans extraction sur disque : seuls les ROIs de
    ce zip sont renvoyés.

    Parameters
    ----------
    zip_path : str
        Chemin vers le fichier RoiSet.zip.
    extract_folder : str, optional
        Obsolète, ignoré (conservé pour compatibilité). Sera retiré dans une
        prochaine version.

    Returns
    -------
    dict
        Dictionnaire {roi_name: roi_data} issu de read_roi.
    """
    if extract_folder is not None:
        warnings.warn(
            "load_rois_from_zip(extract_folder=...) est obsolète et ignoré : "
            "le zip est lu directement.",
            DeprecationWarning,
            stacklevel=2,
        )
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"Fichier ROI introuvable : {zip_path}")

    roi_data = dict(read_roi_zip(zip_path))
    print(f"✅ {len(roi_data)} ROIs chargés depuis {zip_path}")
    return roi_data


def _roi_pixels(roi: dict, hw: tuple[int, int]):
    """
    Pixels intérieurs et contour d'un ROI Fiji.

    Returns
    -------
    tuple ((rr, cc), (rr_p, cc_p)) ou None si le type de ROI n'a pas de surface.
    """
    roi_type = roi.get("type")

    if roi_type in POLYGON_TYPES or (roi_type is None and "x" in roi and "y" in roi):
        x, y = np.asarray(roi["x"], dtype=float), np.asarray(roi["y"], dtype=float)
        return polygon(y, x, shape=hw), polygon_perimeter(y, x, shape=hw)

    if roi_type in BOX_TYPES:
        top, left = roi["top"], roi["left"]
        h, w = roi["height"], roi["width"]
        if roi_type == "rectangle":
            # Pixels left..left+w-1 / top..top+h-1, comme dans Fiji
            x = np.array([left, left + w - 1, left + w - 1, left], dtype=float)
            y = np.array([top, top, top + h - 1, top + h - 1], dtype=float)
            return polygon(y, x, shape=hw), polygon_perimeter(y, x, shape=hw)
        r0, c0 = top + h / 2 - 0.5, left + w / 2 - 0.5
        fill = ellipse(r0, c0, h / 2, w / 2, shape=hw)
        contour = ellipse_perimeter(
            int(round(r0)), int(round(c0)),
            max(int(round(h / 2 - 0.5)), 0), max(int(round(w / 2 - 0.5)), 0),
            shape=hw,
        )
        return fill, contour

    return None


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

    # Géométrie calculée une seule fois par ROI
    shapes, skipped = [], []
    for name, roi in roi_data.items():
        px = _roi_pixels(roi, hw)
        if px is None:
            skipped.append(f"{name} ({roi.get('type')})")
        else:
            shapes.append(px)
    if skipped:
        warnings.warn(
            f"{len(skipped)} ROI(s) sans surface ou de type non pris en charge ignoré(s) : "
            + ", ".join(skipped[:10]) + (" …" if len(skipped) > 10 else ""),
            stacklevel=2,
        )

    # --- Masque 1 : couleurs aléatoires ---
    masked = np.zeros((*hw, 3), dtype=np.uint8)
    for (rr, cc), _ in shapes:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        masked[rr, cc] = color
    path = os.path.join(output_dir, "mask_1.tif")
    tifffile.imwrite(path, masked, photometric="rgb")
    paths["mask_1"] = path

    # --- Masque 2 : couleurs RGB cycliques ---
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    masked = np.zeros((*hw, 3), dtype=np.uint8)
    for idx, ((rr, cc), _) in enumerate(shapes):
        masked[rr, cc] = colors[idx % len(colors)]
    path = os.path.join(output_dir, "mask_2.tif")
    tifffile.imwrite(path, masked, photometric="rgb")
    paths["mask_2"] = path

    # --- Masque 3 : niveaux de gris cumulatifs ---
    # Calcul en uint16 puis plafonnement à 255 (en uint8, 230 + 50 donnait 24)
    acc = np.zeros(hw, dtype=np.uint16)
    for (rr, cc), _ in shapes:
        acc[rr, cc] += 50
    gray = np.minimum(acc, 255).astype(np.uint8)
    path = os.path.join(output_dir, "mask_3.tif")
    tifffile.imwrite(path, gray, photometric="minisblack")
    paths["mask_3"] = path

    # --- Masque 4 : niveaux de gris + contours ---
    acc = np.zeros(hw, dtype=np.uint16)
    for (rr, cc), (rr_p, cc_p) in shapes:
        acc[rr, cc] += 50
        acc[rr, cc] = np.minimum(acc[rr, cc], 255)
        acc[rr_p, cc_p] = 0
    gray = acc.astype(np.uint8)
    path = os.path.join(output_dir, "mask_4.tif")
    tifffile.imwrite(path, gray, photometric="minisblack")
    paths["mask_4"] = path

    print(f"✅ 4 masques générés dans : {output_dir}")
    return paths