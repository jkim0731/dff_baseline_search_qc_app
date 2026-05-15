"""Recipe-driven baseline fitting (sweep runner).

This package may live in several locations. To make all copies work without
each caller managing sys.path, the import-time bootstrap first walks up the
directory tree (up to 5 levels) looking for ``baseline_fitting.py``, then
falls back to a set of well-known paths used in the CodeOcean environment.
"""

import sys
from pathlib import Path

_FALLBACK_PATHS = [
    Path("/code"),
    Path("/root/capsule/code"),
]


def _bootstrap_code_dir() -> None:
    p = Path(__file__).resolve().parent
    for _ in range(5):
        p = p.parent
        if (p / "baseline_fitting.py").exists():
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
            return
    for fb in _FALLBACK_PATHS:
        if (fb / "baseline_fitting.py").exists():
            if str(fb) not in sys.path:
                sys.path.insert(0, str(fb))
            return


_bootstrap_code_dir()
del _bootstrap_code_dir

from .recipe import Recipe  # noqa: E402

__all__ = ["Recipe"]
