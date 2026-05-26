"""Entry point for the binit0 noise-criterion QC app.

After `pip install -e .` in the parent directory, launch with:
    dff-qc
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
    p.add_argument("--roi_list", type=Path, default=None,
                   help="CSV with columns session_key,roi_index to restrict navigation. "
                        "Space/Save+Next walks this list; J/K still work per-session.")
    args = p.parse_args()
    run(runs_dir=args.runs_dir, output=args.output, roi_list=args.roi_list)


if __name__ == "__main__":
    main()
