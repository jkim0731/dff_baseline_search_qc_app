# dFF Baseline Search QC App

Interactive PyQt5 + pyqtgraph app for curating fluorescence baseline fits.

## What it does

Displays, per ROI:
- Corrected F with four baseline estimates (short-window, long-window, F0trend, F0)
- dFF traces (short/long in color; F0trend/F0 as black dashed lines)
- Zoomed ROI image or full FOV with mask overlay, adjustable contrast
- Five metric distributions across all sessions with the current ROI marked

Keyboard shortcuts: `J` prev · `K` next · `S` save · `Space` save + next

## Data layout

```
<parent_dir>/
  <subject>_<YYYY-MM-DD>/
    F_all_array.npy
    baseline_short_window_all_array.npy
    baseline_long_window_all_array.npy
    F0_all.npy
    F0trend_all.npy
    dff_short_window_all_array.npy
    dff_long_window_all_array.npy
    timestamps.npy
    F_noise.npy  F_snr.npy  F_skewness.npy
    bleaching_metric.npy  sustained_metric.npy
    sczdrift_df_all.csv   # plane_id, cell_roi_id per ROI

<data_dir>/
  multiplane-ophys_<subject>_<date>_*_processed_*/
    <plane_id>/extraction/<plane_id>_extraction.h5
      maxImg          — FOV
      rois/coords, data, shape  — sparse pixel masks
```

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py --parent-dir /path/to/scratch/first_try \
               --data-dir   /path/to/data \
               --output     /path/to/curation.csv
```

All arguments are optional; defaults point to the Code Ocean capsule paths.
