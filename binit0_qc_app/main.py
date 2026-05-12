"""Entry point for the binit0 noise-criterion QC app.

Usage:
    python -m binit0_qc_app.main
    python -m binit0_qc_app.main --runs_dir /results/runs --output /results/binit0_qc_curation.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="binit0 noise-criterion QC app")
    p.add_argument("--inputs_dirs", nargs="*", type=Path, default=None,
                   help="Session input directories (default: 0000_first_try + 804670_inputs)")
    p.add_argument("--runs_dir", type=Path, default=None,
                   help="Runs root directory (default: /results/runs)")
    p.add_argument("--output", type=Path, default=None,
                   help="Curation CSV output path")
    args = p.parse_args()

    from .app import run
    run(
        inputs_dirs=args.inputs_dirs,
        runs_dir=args.runs_dir,
        output=args.output,
    )


if __name__ == "__main__":
    main()
