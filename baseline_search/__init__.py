"""Recipe-driven baseline fitting (sweep runner).

This package lives in two places in the repo:

  * ``code/baseline_search/`` — original location, importable from notebooks
    that already have ``code/`` on ``sys.path``.
  * ``code/dff_baseline_search_app/baseline_search/`` — a sibling-of-GUI copy
    so that "run a fit" and "QC the fit" share one folder.

To make both copies work without each caller having to manage sys.path, the
import-time bootstrap walks up the directory tree until it finds
``baseline_fitting.py`` (the math module — kept at the repo root) and inserts
that directory onto ``sys.path``. Up to 5 ancestor levels are searched.
"""

import sys
from pathlib import Path


def _bootstrap_code_dir() -> None:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        p = p.parent
        if (p / "baseline_fitting.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return


_bootstrap_code_dir()
del _bootstrap_code_dir

from .recipe import Recipe  # noqa: E402

__all__ = ["Recipe"]
