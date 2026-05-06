"""Entry point: python main.py [--parent-dir ...] [--data-dir ...] [--output ...]"""

import argparse
from pathlib import Path

from qc_app.app import run
from qc_app.data import DEFAULT_DATA_DIR, DEFAULT_PARENT_DIR
from qc_app.curation import DEFAULT_PATH


def _parse():
    p = argparse.ArgumentParser(description="dFF Baseline QC App")
    p.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR,
                   help="Directory containing per-session subdirectories")
    p.add_argument("--data-dir",   type=Path, default=DEFAULT_DATA_DIR,
                   help="Root of processed ophys data (for FOV / masks)")
    p.add_argument("--output",     type=Path, default=DEFAULT_PATH,
                   help="Path to write/read curation CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(parent_dir=args.parent_dir, data_dir=args.data_dir, output=args.output)
