"""Entry point: python main.py [--parent-dir ...] [--output ...]

Omit --parent-dir to get a folder-chooser dialog on startup.
"""

import argparse
from pathlib import Path

from qc_app.app import run
from qc_app.data import DEFAULT_DATA_DIR


def _parse():
    p = argparse.ArgumentParser(description="dFF Baseline QC App")
    p.add_argument("--parent-dir", type=Path, default=None,
                   help="Directory containing per-session subdirectories "
                        "(opens a dialog if omitted)")
    p.add_argument("--data-dir",   type=Path, default=DEFAULT_DATA_DIR,
                   help="Root of processed ophys data (unused if assets are "
                        "in the session folder)")
    p.add_argument("--output",     type=Path, default=None,
                   help="Path for curation CSV "
                        "(defaults to <parent-dir>/curation.csv)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(parent_dir=args.parent_dir, data_dir=args.data_dir, output=args.output)
