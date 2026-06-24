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


# ── Tests masks ─────────────────────────────────────────────────────────────

def test_load_image_missing():
    from macsima_qc.masks import load_image
    with pytest.raises(FileNotFoundError):
        load_image("fichier_inexistant.tif")


def test_generate_masks_shapes():
    """Vérifie que generate_masks produit 4 fichiers .tif."""
    from macsima_qc.masks import generate_masks
    import cv2

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