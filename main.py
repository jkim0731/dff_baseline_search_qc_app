"""Entry point: ``python main.py`` — no flags needed.

On startup the app asks you (via folder pickers) for:
  parent_dir   (per-session inputs: F, baselines, metrics)
  runs_dir     (parent of NNNN_<slug>/ + index.csv; Cancel to skip compare mode)

CLI flags (--parent-dir, --runs-dir, --output) are escape hatches — they
suppress the corresponding prompt.
"""

import argparse
from pathlib import Path

from qc_app.app import run
from qc_app.data import DEFAULT_DATA_DIR


def _parse():
    p = argparse.ArgumentParser(description="dFF Baseline QC App")
    p.add_argument("--parent-dir", type=Path, default=None,
                   help="Override the auto-detected per-session input directory.")
    p.add_argument("--data-dir",   type=Path, default=DEFAULT_DATA_DIR,
                   help="Root of processed ophys data (unused if assets are "
                        "in the session folder)")
    p.add_argument("--output",     type=Path, default=None,
                   help="Path for curation CSV "
                        "(defaults to <parent-dir>/curation.csv)")
    p.add_argument("--runs-dir",   type=Path, default=None, action="append",
                   help="Pass once or multiple times to seed the compare panel "
                        "with that many runs folders. Suppresses the runs picker.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(parent_dir=args.parent_dir, data_dir=args.data_dir, output=args.output,
        runs_dirs=args.runs_dir)
