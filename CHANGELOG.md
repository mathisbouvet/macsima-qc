# Changelog

## 0.2.0

Methodological changes — **results differ from 0.1.x**.

### Changed
- **Mann-Whitney suggestions** (`run_mann_whitney`, `run_qc_pipeline`)
  - Effect size reported for each feature (Cliff's delta, with magnitude).
  - p-values corrected for multiple testing (Benjamini-Hochberg q-values).
  - A MACSiQView suggestion is emitted only if `q < alpha` **and** `|δ| ≥ min_effect`
    (defaults 0.01 and 0.33). Previously any `p < 0.01` produced a suggestion, which
    with thousands of cells flagged almost every feature.
  - Output columns: medians instead of means, signed relative difference
    (`+12.3 %` / `-8.1 %`), raw p-values (no rounding to 0), sorted by |δ|.
- **KS comparison** (`compute_ks_distances`, `run_comparison`)
  - `centroid_x` / `centroid_y` excluded from the mean KS distance (new `KS_FEATURES`,
    overridable with `features=`).
  - KS distances computed on raw values (KS is rank-based, results are identical)
    and exported to `ks_distances.csv`.
- **KDE plots** (`plot_kde`): no more independent per-dataset rescaling; curves are
  plotted on the manual reference scale, so size shifts between automatic and manual
  segmentations remain visible. Any number of features can be plotted.
- **Figures**: all plotting functions accept `show=False` (save and close, no display);
  `plot_ks_barplot` and `plot_kde` return the figure; the dark theme no longer leaks
  into the user's global matplotlib style.

## 0.1.6

### Fixed
- Built-in reference CSV now shipped with the package (`run_qc_pipeline` works without `ref_path`).
- Output CSVs written to `output_dir` instead of the current directory.
- Cells with missing values kept in the output (`Segmentation_OK = NaN`), same row order as input.
- Mask overflow when more than 5 ROIs overlap (`mask_3` / `mask_4`).
- `load_rois_from_zip` reads the zip directly (no mixing between RoiSets).
- Rectangle and oval Fiji ROIs supported.
- `run_mann_whitney` no longer crashes when all cells are KO or with custom features.
- `__version__` read from package metadata.

### Changed
- OpenCV dependency removed (scikit-image / tifffile used instead).
- The library no longer silences all Python warnings.
