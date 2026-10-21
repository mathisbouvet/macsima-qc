"""
tests/test_macsima_qc.py
------------------------
Tests unitaires pour le package macsima_qc.
Utilise des données synthétiques pour ne pas dépendre de fichiers réels.
"""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest

# ── Helpers pour générer des données synthétiques ──────────────────────────

def make_manual_df(n=100, seed=0):
    """DataFrame de segmentation manuelle avec les 6 colonnes normalisées."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "area":          rng.normal(500, 50, n),
        "perimeter":     rng.normal(80,  8,  n),
        "centroid_x":    rng.uniform(0, 1024, n),
        "centroid_y":    rng.uniform(0, 1024, n),
        "feret":         rng.normal(30,  3,  n),
        "mean_intensity": rng.normal(128, 20, n),
    })

def make_auto_df(n=120, seed=1):
    """DataFrame de segmentation automatique (légèrement différent)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "area":          rng.normal(510, 60, n),
        "perimeter":     rng.normal(82,  9,  n),
        "centroid_x":    rng.uniform(0, 1024, n),
        "centroid_y":    rng.uniform(0, 1024, n),
        "feret":         rng.normal(31,  4,  n),
        "mean_intensity": rng.normal(130, 22, n),
    })

def make_macsiq_df(n=200, seed=2, label="ref"):
    """DataFrame avec les 14 features MACSiQView."""
    from macsima_qc.utils import FEATURES_TO_USE
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(10, 2, n) for f in FEATURES_TO_USE}
    return pd.DataFrame(data)


# ── Tests utils ─────────────────────────────────────────────────────────────

def test_constants_exist():
    from macsima_qc.utils import FEATURES_TO_USE, PARAM_MAP, MANUAL_COLUMNS
    assert len(FEATURES_TO_USE) == 14
    assert len(MANUAL_COLUMNS) == 6
    assert "Nucleus Size" in PARAM_MAP

    def test_version_matches_metadata():
        from importlib.metadata import version
        import macsima_qc
        assert macsima_qc.__version__ == version("macsima-qc")


# ── Tests masks ─────────────────────────────────────────────────────────────

def test_load_image_missing():
    from macsima_qc.masks import load_image
    with pytest.raises(FileNotFoundError):
        load_image("fichier_inexistant.tif")


def test_generate_masks_shapes():
    """Vérifie que generate_masks produit 4 fichiers .tif."""
    from macsima_qc.masks import generate_masks


    image = np.zeros((64, 64, 3), dtype=np.uint8)
    roi_data = {
        "roi_1": {"x": [10, 20, 20, 10], "y": [10, 10, 20, 20]},
        "roi_2": {"x": [30, 50, 50, 30], "y": [30, 30, 50, 50]},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_masks(image, roi_data, output_dir=tmpdir)
        assert len(paths) == 4
        for key, path in paths.items():
            assert os.path.exists(path), f"Fichier manquant : {path}"


@pytest.mark.parametrize("shape,dtype", [
    ((64, 64), np.uint16),       # DAPI 16 bits niveaux de gris
    ((64, 64, 3), np.uint8),     # RGB
    ((2, 64, 64), np.uint16),    # multicanal (C, H, W)
])
def test_generate_masks_any_input(shape, dtype, tmp_path):
    """Masques 8 bits de taille H × W, quel que soit le format de l'image source."""
    import tifffile
    from macsima_qc.masks import generate_masks, load_image

    src = tmp_path / "dapi.tif"
    tifffile.imwrite(src, np.zeros(shape, dtype=dtype))
    image = load_image(str(src))

    roi_data = {"roi_1": {"x": [10, 20, 20, 10], "y": [10, 10, 20, 20]}}
    paths = generate_masks(image, roi_data, output_dir=str(tmp_path / "masks"))

    for name in ("mask_1", "mask_2"):
        m = tifffile.imread(paths[name])
        assert m.shape == (64, 64, 3) and m.dtype == np.uint8
    for name in ("mask_3", "mask_4"):
        m = tifffile.imread(paths[name])
        assert m.shape == (64, 64) and m.dtype == np.uint8
    assert tifffile.imread(paths["mask_3"])[15, 15] == 50


# Encodeur minimal du format ROI ImageJ, pour tester sur de vrais RoiSet.zip
_ROI_TYPES = {"polygon": 0, "rect": 1, "oval": 2, "line": 3, "freehand": 7}


def _roi_bytes(kind, top, left, bottom, right, xs=(), ys=()):
    import struct
    h = bytearray(64)
    h[0:4] = b"Iout"
    struct.pack_into(">h", h, 4, 227)
    h[6] = _ROI_TYPES[kind]
    struct.pack_into(">hhhhh", h, 8, top, left, bottom, right, len(xs))
    body = b"".join(struct.pack(">h", x - left) for x in xs)
    body += b"".join(struct.pack(">h", y - top) for y in ys)
    struct.pack_into(">i", h, 60, 64 + len(body))  # offset du header2, comme Fiji
    return bytes(h) + body + bytes(64)


def _write_roiset(path, rois):
    import zipfile
    with zipfile.ZipFile(path, "w") as z:
        for name, b in rois.items():
            z.writestr(f"{name}.roi", b)
    return str(path)


def test_load_rois_from_zip_reads_only_this_zip(tmp_path, monkeypatch):
    """Deux zips successifs : pas de mélange, rien d'extrait sur disque."""
    from macsima_qc.masks import load_rois_from_zip

    monkeypatch.chdir(tmp_path)
    sq = _roi_bytes("polygon", 10, 10, 21, 21, [10, 20, 20, 10], [10, 10, 20, 20])
    zip_a = _write_roiset(tmp_path / "A.zip", {"a1": sq, "a2": sq, "a3": sq})
    zip_b = _write_roiset(tmp_path / "B.zip", {"b1": sq})

    assert set(load_rois_from_zip(zip_a)) == {"a1", "a2", "a3"}
    assert set(load_rois_from_zip(zip_b)) == {"b1"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["A.zip", "B.zip"]


def test_load_rois_from_zip_missing():
    from macsima_qc.masks import load_rois_from_zip
    with pytest.raises(FileNotFoundError):
        load_rois_from_zip("RoiSet_inexistant.zip")


def test_generate_masks_rectangle_oval_line(tmp_path):
    """Rectangles et ovales Fiji sont dessinés ; les lignes sont ignorées avec un warning."""
    import tifffile
    from macsima_qc.masks import generate_masks, load_rois_from_zip

    zip_path = _write_roiset(tmp_path / "RoiSet.zip", {
        "rect": _roi_bytes("rect", 30, 5, 40, 25),    # 20 × 10 px
        "oval": _roi_bytes("oval", 40, 40, 60, 60),   # disque de diamètre 20 px
        "line": _roi_bytes("line", 0, 0, 5, 5),
    })
    rois = load_rois_from_zip(zip_path)

    with pytest.warns(UserWarning, match="line"):
        paths = generate_masks(np.zeros((64, 64), np.uint16), rois, output_dir=str(tmp_path))

    m3 = tifffile.imread(paths["mask_3"])
    rect_px = int((m3[30:40, 5:25] == 50).sum())
    assert rect_px == 200                                   # exactement left..left+w-1 × top..top+h-1
    assert (m3[:30, :] == 0).all() and (m3[40:, :30] == 0).all()
    oval_px = int((m3[40:60, 40:60] == 50).sum())
    assert abs(oval_px - np.pi * 10 ** 2) < 0.1 * np.pi * 10 ** 2
    assert m3[50, 50] == 50 and m3[40, 40] == 0             # centre plein, coin vide

    m4 = tifffile.imread(paths["mask_4"])
    assert m4[30, 10] == 0 and m4[35, 15] == 50             # contour noir, intérieur gris


def test_mask_overlap_saturates_at_255(tmp_path):
    """6 ROIs superposés : 255 (plafonné), et non 44 (débordement uint8)."""
    import tifffile
    from macsima_qc.masks import generate_masks

    square = {"x": [10, 20, 20, 10], "y": [10, 10, 20, 20]}
    rois = {f"roi_{i}": dict(square) for i in range(6)}
    paths = generate_masks(np.zeros((32, 32), np.uint8), rois, output_dir=str(tmp_path))

    assert tifffile.imread(paths["mask_3"])[15, 15] == 255
    assert tifffile.imread(paths["mask_4"])[15, 15] == 255


# ── Tests comparison ─────────────────────────────────────────────────────────

def test_normalize_datasets():
    from macsima_qc.comparison import normalize_datasets
    df_manual = make_manual_df()
    df_autos = [make_auto_df(), make_auto_df(seed=3)]

    df_m_n, df_autos_n = normalize_datasets(df_manual, df_autos)

    # Après fit_transform sur manual : min=0, max=1
    assert df_m_n["area"].min() >= -0.01
    assert df_m_n["area"].max() <= 1.01
    # Les autos peuvent dépasser [0,1] (transform sans refit)
    assert len(df_autos_n) == 2


def test_compute_ks_distances():
    from macsima_qc.comparison import normalize_datasets, compute_ks_distances
    df_manual = make_manual_df()
    df_autos = [make_auto_df(seed=1), make_auto_df(seed=2)]
    legend = {"Auto1": "Mask A", "Auto2": "Mask B"}

    df_m_n, df_autos_n = normalize_datasets(df_manual, df_autos)
    distances_df = compute_ks_distances(df_m_n, df_autos_n, legend)

    assert "Mean Distance" in distances_df.columns
    assert len(distances_df) == 2
    assert (distances_df["Mean Distance"] >= 0).all()


def test_run_comparison_with_csv(tmp_path):
    from macsima_qc.utils import MANUAL_RENAME_MAP, AUTO_RENAME_MAP, MANUAL_COLUMNS
    from macsima_qc.comparison import run_comparison

    # Créer CSV manuel avec colonnes Fiji
    df_m = make_manual_df(n=80)
    df_m_raw = df_m.rename(columns={v: k for k, v in MANUAL_RENAME_MAP.items()})
    manual_csv = str(tmp_path / "manual.csv")
    df_m_raw.to_csv(manual_csv, index=False)

    # Créer CSVs auto avec colonnes MACSiQView
    auto_csvs = []
    for i in range(2):
        df_a = make_auto_df(n=90, seed=i+1)
        df_a_raw = df_a.rename(columns={v: k for k, v in AUTO_RENAME_MAP.items()})
        path = str(tmp_path / f"auto_{i}.csv")
        df_a_raw.to_csv(path, index=False)
        auto_csvs.append(path)

    import matplotlib
    matplotlib.use("Agg")  # pas d'écran en test

    distances_df = run_comparison(
        manual_path=manual_csv,
        auto_paths=auto_csvs,
        auto_names=["Mask A", "Mask B"],
        output_dir=str(tmp_path / "figures"),
    )
    assert len(distances_df) == 2


# ── Tests qc ────────────────────────────────────────────────────────────────

def test_load_qc_data(tmp_path):
    from macsima_qc.qc import load_qc_data
    from macsima_qc.utils import FEATURES_TO_USE

    df_ref = make_macsiq_df(n=150, seed=0)
    df_test = make_macsiq_df(n=200, seed=1)
    ref_csv = str(tmp_path / "ref.csv")
    test_csv = str(tmp_path / "test.csv")
    df_ref.to_csv(ref_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Signature réelle : load_qc_data(test_path, ref_path, features)
    X_ref_s, X_test_s, df_test_clean, scaler, feats = load_qc_data(
        test_path=test_csv, ref_path=ref_csv
    )

    assert X_ref_s.shape == (150, 14)
    assert X_test_s.shape == (200, 14)
    assert len(feats) == 14


def test_run_isolation_forest(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from macsima_qc.qc import load_qc_data, run_isolation_forest

    df_ref = make_macsiq_df(n=150)
    df_test = make_macsiq_df(n=200, seed=5)
    ref_csv = str(tmp_path / "ref.csv")
    test_csv = str(tmp_path / "test.csv")
    df_ref.to_csv(ref_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    X_ref_s, X_test_s, df_test_clean, _, _ = load_qc_data(
        test_path=test_csv, ref_path=ref_csv
    )
    export = str(tmp_path / "annotated.csv")
    df_out = run_isolation_forest(
        X_ref_s, X_test_s, df_test_clean,
        contamination=0.10,
        output_dir=str(tmp_path / "figures"),
        export_path=export,
    )

    assert "Segmentation_OK" in df_out.columns
    assert "Anomaly_Score" in df_out.columns
    assert set(df_out["Segmentation_OK"].unique()).issubset({1, -1})
    assert os.path.exists(export)


def test_run_mann_whitney(tmp_path):
    from macsima_qc.qc import load_qc_data, run_isolation_forest, run_mann_whitney

    import matplotlib
    matplotlib.use("Agg")

    df_ref = make_macsiq_df(n=150)
    df_test = make_macsiq_df(n=200, seed=5)
    ref_csv = str(tmp_path / "ref.csv")
    test_csv = str(tmp_path / "test.csv")
    df_ref.to_csv(ref_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    X_ref_s, X_test_s, df_test_clean, _, feats = load_qc_data(
        test_path=test_csv, ref_path=ref_csv
    )
    df_annotated = run_isolation_forest(
        X_ref_s, X_test_s, df_test_clean,
        output_dir=str(tmp_path / "figures"),
        export_path=str(tmp_path / "annotated.csv"),
    )

    export = str(tmp_path / "suggestions.csv")
    summary_df = run_mann_whitney(df_annotated, feats, export_path=export)

    assert "Variable" in summary_df.columns
    assert "p-value" in summary_df.columns
    assert os.path.exists(export)


def test_run_qc_pipeline(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from macsima_qc.qc import run_qc_pipeline

    df_ref = make_macsiq_df(n=150)
    df_test = make_macsiq_df(n=200, seed=7)
    ref_csv = str(tmp_path / "ref.csv")
    test_csv = str(tmp_path / "test.csv")
    df_ref.to_csv(ref_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    # Signature réelle : run_qc_pipeline(test_path, ref_path, ...)
    df_annotated, df_suggestions = run_qc_pipeline(
        test_path=test_csv,
        ref_path=ref_csv,
        contamination=0.10,
        output_dir=str(tmp_path / "figures"),
    )

    assert len(df_annotated) == 200
    assert len(df_suggestions) > 0

# ── Tests référence intégrée ───────────────────────────────────────────────────

def test_builtin_reference_is_packaged():
    """Le CSV de référence doit être livré avec le package (wheel + sdist)."""
    from importlib import resources
    from macsima_qc.qc import BUILTIN_REFERENCE
    from macsima_qc.utils import FEATURES_TO_USE

    ref = resources.files("macsima_qc.data").joinpath(BUILTIN_REFERENCE)
    assert ref.is_file(), f"{BUILTIN_REFERENCE} absent de macsima_qc/data/"

    with ref.open("r") as f:
        df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    assert set(FEATURES_TO_USE).issubset(df.columns)
    assert len(df) > 1000


def test_run_qc_pipeline_builtin_reference(tmp_path):
    """Usage minimal du README : aucun ref_path."""
    import matplotlib
    matplotlib.use("Agg")
    from macsima_qc.qc import run_qc_pipeline

    from macsima_qc.qc import _load_builtin_reference
    from macsima_qc.utils import FEATURES_TO_USE

    # Données test réalistes : un échantillon de la référence + quelques cellules aberrantes
    ref = _load_builtin_reference(FEATURES_TO_USE)
    df_test = ref.sample(n=300, random_state=0).reset_index(drop=True)
    df_test.loc[:29, ["Cell Size", "Nucleus Size"]] *= 4
    test_csv = str(tmp_path / "test.csv")
    df_test.to_csv(test_csv, index=False)

    df_annotated, df_suggestions = run_qc_pipeline(
        test_path=test_csv,
        output_dir=str(tmp_path / "figures"),
    )
    assert len(df_annotated) == 300
    assert (df_annotated["Segmentation_OK"] == 1).mean() > 0.5
    assert len(df_suggestions) > 0


def test_run_mann_whitney_all_ko(tmp_path):
    """Toutes les cellules KO : pas de crash, tableau vide avec les bonnes colonnes."""
    from macsima_qc.qc import run_mann_whitney
    from macsima_qc.utils import FEATURES_TO_USE

    df = make_macsiq_df(n=50)
    df["Segmentation_OK"] = -1
    summary_df = run_mann_whitney(df, export_path=str(tmp_path / "s.csv"))
    assert summary_df.empty
    assert "p-value" in summary_df.columns


def test_run_mann_whitney_custom_feature(tmp_path):
    """Une feature absente de PARAM_MAP ne doit pas provoquer de KeyError."""
    from macsima_qc.qc import run_mann_whitney
    from macsima_qc.utils import FEATURES_TO_USE

    df = make_macsiq_df(n=200)
    df["Extra"] = np.r_[np.full(100, 1.0), np.full(100, 9.0)] + np.random.default_rng(0).normal(0, .1, 200)
    df["Segmentation_OK"] = np.r_[np.ones(100), -np.ones(100)].astype(int)
    summary_df = run_mann_whitney(df, FEATURES_TO_USE + ["Extra"], export_path=str(tmp_path / "s.csv"))
    assert "Extra" in summary_df["Variable"].values


def test_outputs_go_to_output_dir(tmp_path, monkeypatch):
    """Aucun fichier ne doit être écrit dans le dossier courant."""
    import matplotlib
    matplotlib.use("Agg")
    from macsima_qc.qc import run_qc_pipeline

    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)

    ref_csv = str(tmp_path / "ref.csv")
    test_csv = str(tmp_path / "test.csv")
    make_macsiq_df(n=150).to_csv(ref_csv, index=False)
    make_macsiq_df(n=200, seed=7).to_csv(test_csv, index=False)

    out = tmp_path / "results"
    run_qc_pipeline(test_path=test_csv, ref_path=ref_csv, output_dir=str(out))

    assert list(work.iterdir()) == []
    assert (out / "segmentation_test_annotated.csv").exists()
    assert (out / "macsiq_param_suggestions.csv").exists()


def test_pipeline_keeps_rows_with_nan(tmp_path):
    """Les cellules avec NaN restent dans la sortie (non évaluées), dans l'ordre d'origine."""
    import matplotlib
    matplotlib.use("Agg")
    from macsima_qc.qc import run_qc_pipeline

    ref_csv = str(tmp_path / "ref.csv")
    make_macsiq_df(n=150).to_csv(ref_csv, index=False)

    df_test = make_macsiq_df(n=200, seed=7)
    df_test["Cell ID"] = range(200)
    df_test.loc[[3, 50, 199], "Cell Size"] = np.nan
    test_csv = str(tmp_path / "test.csv")
    df_test.to_csv(test_csv, index=False)

    df_annotated, _ = run_qc_pipeline(
        test_path=test_csv, ref_path=ref_csv, output_dir=str(tmp_path / "out"),
    )

    assert len(df_annotated) == 200
    assert df_annotated["Cell ID"].tolist() == list(range(200))
    assert df_annotated["Segmentation_OK"].isna().sum() == 3
    assert df_annotated.loc[[3, 50, 199], "Segmentation_OK"].isna().all()
    assert set(df_annotated["Segmentation_OK"].dropna().unique()).issubset({1, -1})

    exported = pd.read_csv(tmp_path / "out" / "segmentation_test_annotated.csv")
    assert len(exported) == 200