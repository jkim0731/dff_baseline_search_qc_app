"""Entry point: ``python main.py`` — no flags needed.

On startup the app shows a single folder picker:

  inputs_dir   the folder that contains per-session subfolders
               (each subfolder has F_all_array.npy, timestamps.npy, etc.)

The **parent** of the chosen inputs folder is automatically treated as the
runs root (it must contain index.csv and NNNN_<slug>/ run folders).

Example layout::

    /results/runs/               ← runs root (auto-detected as parent)
        index.csv
        0001_first_try/
        0002_cpos3_cneg3_lowess/
        ...
        first_try/               ← pick THIS as the inputs folder
            755252_2024-11-12/
                F_all_array.npy
                ...
            755252_2024-11-19/
                ...

CLI flags (--parent-dir, --output) are escape hatches that suppress the
corresponding prompt.
"""

import argparse
from pathlib import Path

from qc_app.app import run
from qc_app.data import DEFAULT_DATA_DIR


def _parse():
    p = argparse.ArgumentParser(description="dFF Baseline QC App")
    p.add_argument("--parent-dir", type=Path, default=None,
                   help="Inputs folder (parent of session subfolders). "
                        "Suppresses the folder picker.")
    p.add_argument("--data-dir",   type=Path, default=DEFAULT_DATA_DIR,
                   help="Root of processed ophys data (unused if assets are "
                        "in the session folder).")
    p.add_argument("--output",     type=Path, default=None,
                   help="Path for curation CSV "
                        "(defaults to <parent-dir>/curation.csv).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(parent_dir=args.parent_dir, data_dir=args.data_dir, output=args.output)
