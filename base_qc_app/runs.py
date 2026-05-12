"""Discovery and loading of baseline_search numbered run folders.

A "run" is one parameter set applied across one or more sessions, written by
``baseline_search/run.py`` (co-located in this app folder, with a mirror at
``code/baseline_search/`` for direct notebook use) to::

    <runs_dir>/
        index.csv                              # one row per run, recipe fields flattened
        NNNN_<slug>/
            recipe.json
            metadata.json
            <session_key>/
                F0trend_all.npy
                F0_all.npy
                ...
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_RUNS_DIR = Path("/results/runs")

# kind → filename within a run/session folder
_RUN_BASELINE_FILES = {
    "F0trend": "F0trend_all.npy",
    "F0":      "F0_all.npy",
}


def discover_runs(runs_dir: Path = DEFAULT_RUNS_DIR) -> pd.DataFrame:
    """Read ``runs_dir/index.csv`` and return a DataFrame ordered by run_id.

    Always returns a DataFrame; if the index is missing or empty, returns an
    empty DataFrame with at least the columns the dialog needs.
    """
    df, _status = discover_runs_with_status(runs_dir)
    return df


def discover_runs_with_status(runs_dir: Path) -> tuple[pd.DataFrame, str]:
    """Return ``(df, human_status)`` so callers can surface diagnostics.

    Status strings:
        - "n runs"                              (success — n is the row count)
        - "missing index.csv (not a runs folder)"
        - "missing index.csv (looks like an inputs folder: <session>/...)"
        - "missing index.csv (looks like a single run folder: <session>/...)"
        - "permission denied"
        - "index.csv unreadable: <error>"
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return _empty_index(), "path does not exist"
    if not runs_dir.is_dir():
        return _empty_index(), "not a directory"
    idx = runs_dir / "index.csv"
    if not idx.exists():
        # Diagnose the folder shape by looking at file contents (not names).
        try:
            subdirs = [p for p in runs_dir.iterdir() if p.is_dir()]
        except PermissionError:
            return _empty_index(), "permission denied"
        if not subdirs:
            hint = "empty folder"
        elif any((d / "recipe.json").exists() and (d / "metadata.json").exists()
                 for d in subdirs):
            hint = "no index.csv — runs found but unindexed (regenerate index)"
        elif any((d / "F_all_array.npy").exists() for d in subdirs):
            hint = "looks like an inputs folder (sessions inside) — pick the runs parent instead"
        elif any((d / "F0trend_all.npy").exists() for d in subdirs):
            hint = "looks like a single run folder — pick its parent"
        else:
            hint = "no index.csv"
        return _empty_index(), hint

    try:
        df = pd.read_csv(idx, dtype={"run_id": str})
    except (PermissionError, OSError) as exc:
        return _empty_index(), f"index.csv unreadable: {exc}"
    df["source_dir"] = str(runs_dir)
    df = df.sort_values("run_id").reset_index(drop=True)
    return df, f"{len(df)} run(s)"


def _empty_index() -> pd.DataFrame:
    return pd.DataFrame(columns=["run_id", "slug", "run_dir",
                                 "created_at", "description", "source_dir"])


def discover_runs_multi(runs_dirs: list[Path]) -> pd.DataFrame:
    """Concatenate indices from multiple runs roots.

    Adds a ``source_dir`` column (string) so duplicate run_ids from different
    sources stay distinguishable. The unique key across the merged frame is
    ``run_dir`` (which is an absolute path).
    """
    df, _status = discover_runs_multi_with_status(runs_dirs)
    return df


def discover_runs_multi_with_status(runs_dirs: list[Path]
                                    ) -> tuple[pd.DataFrame, dict[str, str]]:
    """Like ``discover_runs_multi`` but also returns ``{source_path: status}``."""
    statuses: dict[str, str] = {}
    if not runs_dirs:
        return _empty_index(), statuses
    frames = []
    for d in runs_dirs:
        df_i, status = discover_runs_with_status(Path(d))
        statuses[str(d)] = status
        if not df_i.empty:
            frames.append(df_i)
    if not frames:
        return _empty_index(), statuses
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["run_dir"]).reset_index(drop=True)
    return df, statuses


def list_run_sessions(run_dir: Path) -> list[str]:
    """Session keys (subdir names) present inside one run folder."""
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    return sorted(p.name for p in run_dir.iterdir() if p.is_dir())


@lru_cache(maxsize=64)
def load_run_baseline(run_dir_str: str, session_key: str, kind: str) -> np.ndarray:
    """Memory-mapped (N, T) array for one (run, session, kind)."""
    if kind not in _RUN_BASELINE_FILES:
        raise ValueError(
            f"Unknown kind {kind!r}; expected one of {list(_RUN_BASELINE_FILES)}"
        )
    p = Path(run_dir_str) / session_key / _RUN_BASELINE_FILES[kind]
    if not p.exists():
        raise FileNotFoundError(p)
    return np.load(p, mmap_mode="r")
