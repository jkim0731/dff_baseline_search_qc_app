"""Entry point for the binit0 noise-criterion QC app.

After `pip install -e .` in the parent directory, launch with:
    binit0-qc
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .app import run


def main():
    p = argparse.ArgumentParser(description="binit0 noise-criterion QC app")
    p.add_argument("--runs_dir", type=Path, default=None,
                   help="Directory containing numbered run sub-folders. "
                        "If omitted, a directory-picker dialog is shown at startup.")
    p.add_argument("--output", type=Path, default=None,
                   help="Curation CSV output path "
                        "(default: <runs_dir>/binit0_qc_curation.csv)")
    args = p.parse_args()
    run(runs_dir=args.runs_dir, output=args.output)


if __name__ == "__main__":
    main()
