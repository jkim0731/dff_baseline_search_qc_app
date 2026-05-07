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

## Run a fit

From inside `code/dff_baseline_search_app/`:

```bash
python -m baseline_search.run \
    --recipe baseline_search/recipes/first_try.json \
    --inputs_dir /results/runs/first_try \
    --out /results/runs \
    --slug first_try_replication \
    --sessions 755252_2024-11-19
```

Outputs:

```
/results/runs/0001_first_try_replication/
    recipe.json                      # canonical, validated by Pydantic
    metadata.json                    # created_at, host, git_rev, sessions[]
    755252_2024-11-19/
        F0trend_all.npy   (N, T) float32
        F0_all.npy        (N, T) float64
        res_all.npy       (N, n_params)
        loss_all.npy      (N,)
        info.json
```

A row is appended to `/results/runs/index.csv` with the recipe leaves flattened
(`recipe_sigma_method`, `recipe_M_c_pos`, …). The QC app reads this file.

## QC the fits

```bash
python main.py
```

Picks parent_dir (per-session inputs), then one or more runs folders, then
launches the GUI. The compare-mode dock opens by default and reads the runs
indices from every chosen source.

Compare panel features:
- Multiple sources with per-source diagnostics ("looks like an inputs folder",
  "no index.csv", "2 runs", …) so empty results are explainable at a glance.
- Check the runs you want to compare; the *Differences* table auto-shows only
  recipe parameters that disagree across the checked runs.
- Quick-assign maps the first N checked runs into slots 1..4 with one click.
- When slots are assigned, the session dropdown is restricted to the
  intersection of sessions present in every selected run.

Keyboard: `J` prev · `K` next · `S` save · `Space` save+next · `R` toggle
compare mode · `1`–`4` toggle slot traces.

## Inputs layout

```
<parent_dir>/
  <subject>_<YYYY-MM-DD>/
    F_all_array.npy
    baseline_short_window_all_array.npy
    baseline_long_window_all_array.npy
    F0_all.npy
    F0trend_all.npy
    timestamps.npy
    F_noise.npy  F_snr.npy  F_skewness.npy
    bleaching_metric.npy  sustained_metric.npy
    sczdrift_df_all.csv
    <plane_id>_mean_img.npy  <plane_id>_max_img.npy
    <plane_id>_roi_table.pkl
```

## Install

```bash
pip install -e .
```

This installs both `qc_app` and `baseline_search`, plus console scripts
`dff-qc` (GUI) and `dff-fit` (runner).
