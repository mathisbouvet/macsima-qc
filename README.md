# macsima-qc
[![Tests](https://github.com/mathisbouvet/macsima-qc/actions/workflows/tests.yml/badge.svg)](https://github.com/mathisbouvet/macsima-qc/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/macsima-qc.svg)](https://pypi.org/project/macsima-qc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

**Quality control pipeline for MACSima / MACSiQView cell segmentation**

Python package for evaluating and comparing cell segmentation results from the MACSima cyclic immunofluorescence platform, using manual ROI references from Fiji/ImageJ.


<img src="assets/macsima_qc_home.png" width="500">

## Installation

```bash
pip install macsima-qc
```

Or from source (development mode):

```bash
git clone https://github.com/mathisbouvet/macsima-qc
cd macsima-qc
pip install -e .
```

---

<p align="center">
<img src="assets/home.png" width="500">
</p>


## Workflow

```
MACSiQView segmentation (.csv)
            │
    run_qc_pipeline()  ← Isolation Forest + Mann-Whitney
            │
    ┌───────┴────────────────────────┐
    │ segmentation_test_annotated.csv │  ← OK / KO labels per cell
    │ macsiq_param_suggestions.csv   │  ← MACSiQView adjustment suggestions
    └─────────────────────────────────┘

── Optional ──────────────────────────────────────────────
Fiji ROIs (.zip) ──► generate_masks() ──► run_comparison()  ← KS distances + KDE
```

---

## Quick start

### Minimal usage — built-in reference

No reference file needed. Just point to your MACSiQView segmentation CSV:

```python
from macsima_qc import run_qc_pipeline

df_annotated, df_suggestions = run_qc_pipeline(
    test_path="Segmentation_test.csv",
    contamination=0.10,
    output_dir="figures/",
)
```

### With a custom reference

If you want to use your own manually validated segmentation as reference:

```python
from macsima_qc import run_qc_pipeline

df_annotated, df_suggestions = run_qc_pipeline(
    test_path="Segmentation_test.csv",
    ref_path="my_reference.csv",
    contamination=0.10,
)
```

---

## 1. Exporting data from MACSiQView

### 1.a What data to export

After segmentation in MACSiQView, navigate to the **Feature Table** tab and select the following 14 morphological parameters for export. Only morphological descriptors are used, fluorescence intensities are excluded.

| Parameter | Description |
|-----------|-------------|
| `Cell Bbox X Size` | Bounding box width |
| `Cell Bbox Y Size` | Bounding box height |
| `Cell Shape Circle Like` | Circularity index |
| `Cell Shape Ellipse Like` | Ellipse similarity |
| `Cell Shape Elongation` | Elongation ratio |
| `Cell Shape Square Like` | Square similarity |
| `Cell Shape Triangle Like` | Triangle similarity |
| `Cell Size` | Cell area |
| `Nucleus Size` | Nucleus area |
| `Nucleus Roundness` | Nucleus roundness |
| `Nucleus Convexity` | Nucleus convexity |
| `Cell Convexity` | Cell convexity |
| `Quality Cell In-Focus` | Focus quality score |
| `Quality Nuclear Segmentation` | Nuclear segmentation quality |

> **These 14 features are mandatory.** The pipeline will raise an error if any are missing from the exported CSV.

### 1.b How to export

1. In MACSiQView, open the **Feature Table** panel
2. Select the 14 parameters listed above
3. Export as `.csv`
4. This exported file is your `test_path` input for `run_qc_pipeline()`

### 1.c About the built-in reference

The package ships with a built-in reference segmentation derived from a manually validated MACSima acquisition (DAPI channel, human embryonic tissue). It contains **4,641 cells** described by the same 14 morphological features listed above, exported using the exact same MACSiQView protocol.

> If your tissue type differs significantly from human embryonic tissue (e.g. non-embryonic tissue, very different cell density or morphology), consider providing your own reference via `ref_path`.

### 1.d QC pipeline — Isolation Forest

Before fixing the contamination parameter, the model sensitivity is analyzed across a range of values (0.01 → 0.20). The Isolation Forest is then trained exclusively on the reference data. Each test cell receives a label (`Segmentation_OK`: `1` or `-1`) and a continuous anomaly score.

For each feature, OK and KO cells are then compared with a two-sided Mann-Whitney U test. Because MACSima datasets contain thousands of cells, almost any difference becomes statistically significant, so the pipeline also reports an **effect size** (Cliff's delta, δ ∈ [−1, 1]; δ > 0 means larger values in KO cells) and corrects p-values for the 14 tests (Benjamini-Hochberg q-values).

A MACSiQView adjustment suggestion is emitted only when the difference is both significant **and** large enough: `q < alpha` (default 0.01) and `|δ| ≥ min_effect` (default 0.33, a "medium" effect according to Romano et al., 2006). Both thresholds can be changed in `run_qc_pipeline(..., alpha=0.01, min_effect=0.33)`. Results are exported to `macsiq_param_suggestions.csv`, sorted by effect size.

| Column | Meaning |
|--------|---------|
| `Médiane OK` / `Médiane KO` | Median of the feature in OK / KO cells |
| `Delta de Cliff`, `Taille d'effet` | Effect size and its magnitude (négligeable < 0.147 ≤ faible < 0.33 ≤ moyenne < 0.474 ≤ forte) |
| `Écart KO vs OK (médianes)` | Signed relative difference of the medians |
| `p-value`, `q-value (BH)` | Raw and Benjamini-Hochberg adjusted p-values |
| `Suggestion d'ajustement` | MACSiQView parameter change, only if `q < alpha` and `|δ| ≥ min_effect` |

> All functions that draw a figure accept `show=False` to save it without displaying it (useful in scripts and batch runs).



## Outputs

| File | Description |
|------|-------------|
| `mask_1–4.tif` | ROI masks for MACSiQView import |
| `figures/ks_distances.csv` | KS distance per feature and mean distance per segmentation |
| `figures/ks_average_distances.png` | KS distance barplot per segmentation |
| `figures/kde_comparative_distributions.png` | KDE distribution comparison |
| `figures/contamination_sensitivity.png` | Isolation Forest sensitivity curve |
| `figures/anomaly_scores_distribution.png` | Anomaly score histogram |
| `figures/segmentation_test_annotated.csv` | Cells annotated OK/KO + anomaly score |
| `figures/macsiq_param_suggestions.csv` | MACSiQView parameter adjustment suggestions |

<p align="center">
<img src="assets/anomaly_scores_distribution.png" width="600">
</p>

<p align="center"><em>Anomaly score distribution from the Isolation Forest — cells below the decision threshold are flagged as <code>Segmentation_KO</code>.</em></p>

## 2. Creating a reference segmentation *(optional)*

> For the full step-by-step protocol, see the [01_segmentation_qc](https://github.com/mathisbouvet/MACSima_Advanced-Spatial-Omics-Pipeline/blob/main/protocols/Test%20of%20segmentation.md).

A reference DAPI image is extracted from the MACSima system and exported in `.tif` format for processing in Fiji. Regions of interest (ROI) are manually drawn to serve as the basis for segmentation masks.

- **DAPI Image** : exported in `tiff` format via MACSima
- **RoiSet** : created and exported from Fiji in `zip` format

Once the ROIs are defined, four types of masks are generated and imported into MACSiQView as segmentation inputs:

| File | Description |
|------|-------------|
| `mask_1.tif` | ROIs with random colors, black background |
| `mask_2.tif` | ROIs with cyclic RGB colors (R/G/B), black background |
| `mask_3.tif` | ROIs in cumulative grayscale |
| `mask_4.tif` | ROIs in cumulative grayscale + black contours |




### 2a. Comparing segmentations

```python
from macsima_qc import run_comparison

distances = run_comparison(
    manual_path = "Segmentation_manuelle.csv",
    auto_paths  = [
        "Mask_3_Single_Cell.csv",
        "Mask_3_Tissue.csv",
        "Mask_4_Import_mask.csv",
        "Mask_4_Single_Cell.csv",
        "Mask_4_Tissue.csv",
    ],
    auto_names  = [
        "Mask 3 - Single Cell",
        "Mask 3 - Tissue",
        "Mask 4 - Import Mask",
        "Mask 4 - Single Cell",
        "Mask 4 - Tissue",
    ],
    output_dir = "figures/",
)
```

This produces a barplot of average KS distances and KDE distribution curves for the main morphological parameters, identifying the automatic segmentation closest to the manual reference.

- The mean KS distance is computed on `area`, `perimeter`, `feret` and `mean_intensity`. Cell centroids are loaded but excluded from the score, since cell position depends on the imaged field, not on segmentation quality (change it with `features=[...]`).
- KDE curves are all expressed on the scale of the manual reference (0–1 = manual min–max), so a systematic size shift of an automatic segmentation stays visible.

<p align="center">
<img src="assets/ks_average_distances.png" width="600">
</p>

<p align="center">
<img src="assets/distributions_comparées.png" width="800">
</p>

<p align="center"><em>Lower KS distance = closer match to the manual reference. Here, <strong>Mask 4 – Single Cell</strong> performs best across most morphological parameters.</em></p>

---



## Requirements

- Python ≥ 3.9
- numpy, matplotlib, scikit-image, tifffile, read-roi
- pandas, scikit-learn, scipy, seaborn

---

## Citation

If you use this package in a publication, please cite:

Bouvet M. (2026)

---

## License

MIT
