# dFF Baseline Search App

One folder for **running baseline fits** and **curating them**. Two top-level packages:

- `baseline_search/` — recipe-driven runner. A `recipe.json` declares how `x0`,
  `sigma`, `bounds`, `M`, `model` and `fluctuations` are derived; the runner
  resolves the recipe and applies `fit_baseline` per-ROI, writing
  `F0trend_all.npy`, `F0_all.npy`, etc. to a numbered run folder.
- `qc_app/` — interactive PyQt5 + pyqtgraph viewer. Compares any number of runs
  side-by-side per ROI; auto-shows only recipe parameters that actually differ.

The original `code/baseline_search/` package is preserved for notebooks that
import it directly; the copy here keeps run + QC together.

## Expected folder layout

The app assumes this structure:

```
<runs_root>/               ← the parent; auto-detected from your inputs choice
    index.csv              ← written by the runner; one row per run
    0001_first_try/        ← numbered run folders
    0002_cpos3_cneg3/
    ...
    first_try/             ← pick THIS as the inputs folder in the GUI
        755252_2024-11-12/
            F_all_array.npy
            timestamps.npy
            baseline_short_window_all_array.npy
            baseline_long_window_all_array.npy
            F0trend_all.npy
            F0_all.npy
            F_noise.npy  F_snr.npy  F_skewness.npy
            bleaching_metric.npy  sustained_metric.npy
            sczdrift_df_all.csv          (optional — ROI metadata)
            <plane_id>_mean_img.npy
            <plane_id>_max_img.npy
            <plane_id>_roi_table.pkl
        755252_2024-11-19/
            ...
```

**The inputs folder** is the one that contains per-session subfolders (each
with `F_all_array.npy`). Its **parent** is automatically treated as the runs
root and is scanned for `index.csv` + numbered run folders.

> Example: if you pick `/results/runs/first_try/` as the inputs folder, the
> app will look for runs in `/results/runs/`.

## QC the fits

```bash
cd code/dff_baseline_search_qc_app
python main.py
```

A single folder picker appears — select the **inputs folder** (e.g.
`/results/runs/first_try/`). The app then:

1. Lists all session subfolders found inside it.
2. Automatically scans the parent directory for run folders and populates the
   compare panel.

Compare panel features:
- Check the runs you want to compare; the *Differences* table auto-shows only
  recipe parameters that disagree across the checked runs.
- Quick-assign maps the first N checked runs into slots 1..4 with one click.
- When slots are assigned, the session dropdown is restricted to the
  intersection of sessions present in every selected run.

Keyboard: `J` prev · `K` next · `S` save · `Space` save+next · `R` toggle
compare mode · `1`–`4` toggle slot traces.

## Run a fit

From inside `code/dff_baseline_search_qc_app/`:

```bash
python -m baseline_search.run \
    --recipe baseline_search/recipes/first_try.json \
    --inputs_dir /results/runs/first_try \
    --out /results/runs \
    --slug first_try_replication \
    --sessions 755252_2024-11-19
```

Outputs land in `/results/runs/0001_first_try_replication/` (auto-numbered)
and a row is appended to `/results/runs/index.csv`.

Available recipes in `baseline_search/recipes/`:

| File | c_pos | c_neg | fluctuations |
|------|-------|-------|--------------|
| `first_try.json` | 2 | 3 | lowess |
| `cpos2_cneg4_lowess.json` | 2 | 4 | lowess |
| `cpos2_cneg5_lowess.json` | 2 | 5 | lowess |
| `cpos3_cneg3_lowess.json` | 3 | 3 | lowess |
| `cpos3_cneg4_lowess.json` | 3 | 4 | lowess |
| `cpos3_cneg5_lowess.json` | 3 | 5 | lowess |
| `cpos4_cneg4_lowess.json` | 4 | 4 | lowess |
| `cpos4_cneg5_lowess.json` | 4 | 5 | lowess |
| `percentile_variant.json` | 3 | 3 | percentile |

## Install

```bash
pip install -e .
```

This installs both `qc_app` and `baseline_search`, plus console scripts
`dff-qc` (GUI) and `dff-fit` (runner).
