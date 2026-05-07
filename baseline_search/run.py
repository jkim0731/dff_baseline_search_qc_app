"""CLI runner: resolve a recipe and fit baselines for one or more sessions.

Usage (run from inside ``code/dff_baseline_search_app/``)::

    python -m baseline_search.run \\
        --recipe baseline_search/recipes/first_try.json \\
        --inputs_dir /results/runs/first_try \\
        --out /results/runs \\
        --slug first_try_replication \\
        --sessions 755252_2024-11-19

Outputs::

    /results/runs/0001_first_try_replication/
        recipe.json
        metadata.json
        index.csv (appended at /results/runs/index.csv)
        755252_2024-11-19/
            F0trend_all.npy   (N, T) float32
            F0_all.npy        (N, T) float64
            res_all.npy       (N, n_params)  trend-stage final params
            loss_all.npy      (N,)            per-ROI M-loss summary
            info.json                         per-ROI diagnostics
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from joblib import Parallel, delayed

# Allow `python -m baseline_search.run` *and* a bare-script invocation. The
# package's ``__init__.py`` already adds /code/ to sys.path so ``baseline_fitting``
# is importable; if this module is run as a top-level script, also add its parent.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import baseline_search  # noqa: F401  # triggers the sys.path bootstrap
    from baseline_search.recipe import Recipe  # noqa: E402
    from baseline_search.resolve import ResolvedFit, resolve  # noqa: E402
else:
    from .recipe import Recipe
    from .resolve import ResolvedFit, resolve

from baseline_fitting import fit_baseline  # noqa: E402


# ---------------------------------------------------------------------------
# Per-ROI worker — must be top-level for joblib/loky pickling.
# ---------------------------------------------------------------------------
def _fit_one_roi(
    F_row,
    timestamps,
    model_fn,
    x0,
    bounds,
    M,
    M_fluctuations,
    sigma,
    fit_baseline_kwargs,
):
    """Fit one ROI. Returns (F0trend, F0, params, loss, info_summary)."""
    F0, F0trend, res, info = fit_baseline(
        F_row,
        timestamps,
        model_fn,
        x0,
        bounds=bounds,
        M=M,
        M_fluctuations=M_fluctuations,
        fixed_sigma=sigma,
        **fit_baseline_kwargs,
    )

    # Loss summary — defined for LOWESS only; matches notebook formula:
    #   ls = lowess_sigma * F0trend  (mode=ratio) or lowess_sigma (mode=subtract)
    #   loss = mean(M.rho((F - F0) / ls) * ls**2)
    method = fit_baseline_kwargs.get("method", "lowess")
    mode = fit_baseline_kwargs.get("mode", "ratio")
    if method == "lowess":
        ls_sigma = info["lowess_sigma"]
        if mode == "ratio":
            ls = ls_sigma * F0trend
        else:
            ls = ls_sigma * np.ones_like(F0trend)
        with np.errstate(divide="ignore", invalid="ignore"):
            loss = float(np.mean(M.rho((F_row - F0) / ls) * ls ** 2))
    else:
        loss = float("nan")

    info_summary = {
        "trend": {
            "sigma":   float(getattr(res, "sigma", float("nan"))),
            "nit":     int(getattr(res, "nit", -1)),
            "success": bool(getattr(res, "success", False)),
            "weights_inlier_frac": (
                float(np.mean(np.asarray(res.weights) > 0))
                if hasattr(res, "weights") else None
            ),
        },
        "fluctuations": {
            "method": method,
            **(
                {"lowess_sigma": float(info["lowess_sigma"])}
                if method == "lowess"
                else {
                    "percentile": float(info["percentile"]),
                    "size": int(info["size"]),
                }
            ),
        },
    }

    params = np.asarray(res.x, dtype=np.float64)
    return F0trend, F0, params, loss, info_summary


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------
def _load_session_inputs(inputs_dir: Path, session: str):
    p = inputs_dir / session
    F = np.load(p / "F_all_array.npy")
    timestamps = np.load(p / "timestamps.npy")
    bl_long = (
        np.load(p / "baseline_long_window_all_array.npy")
        if (p / "baseline_long_window_all_array.npy").exists()
        else None
    )
    bl_short = (
        np.load(p / "baseline_short_window_all_array.npy")
        if (p / "baseline_short_window_all_array.npy").exists()
        else None
    )
    return F, timestamps, bl_long, bl_short


def _run_session(
    recipe: Recipe,
    inputs_dir: Path,
    out_dir: Path,
    session: str,
    n_jobs: int,
    verbose: int,
) -> dict:
    F, timestamps, bl_long, bl_short = _load_session_inputs(inputs_dir, session)
    rf: ResolvedFit = resolve(
        recipe,
        F,
        timestamps,
        baseline_long=bl_long,
        baseline_short=bl_short,
    )

    N = F.shape[0]
    sigma_iter = (
        [None] * N if rf.sigma_all is None else [float(s) for s in rf.sigma_all]
    )

    def _job(i):
        return _fit_one_roi(
            F[i],
            rf.timestamps,
            rf.model_fn,
            rf.x0_all[i].tolist(),
            rf.bounds,
            rf.M,
            rf.M_fluctuations,
            sigma_iter[i],
            rf.fit_baseline_kwargs,
        )

    t0 = time.time()
    if n_jobs == 1:
        results = [_job(i) for i in range(N)]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(
            delayed(_job)(i) for i in range(N)
        )
    runtime_s = time.time() - t0

    F0trend_all = np.stack([r[0] for r in results]).astype(np.float32)
    F0_all      = np.stack([r[1] for r in results]).astype(np.float64)
    params_all  = np.stack([r[2] for r in results]).astype(np.float64)
    loss_all    = np.asarray([r[3] for r in results], dtype=np.float64)
    info_per_roi = [r[4] for r in results]

    sess_dir = out_dir / session
    sess_dir.mkdir(parents=True, exist_ok=True)
    np.save(sess_dir / "F0trend_all.npy", F0trend_all)
    np.save(sess_dir / "F0_all.npy",      F0_all)
    np.save(sess_dir / "res_all.npy",     params_all)
    np.save(sess_dir / "loss_all.npy",    loss_all)
    with open(sess_dir / "info.json", "w") as f:
        json.dump(
            {
                "param_names": rf.model_param_names,
                "per_roi": info_per_roi,
            },
            f,
            indent=2,
        )

    return {
        "session":   session,
        "n_rois":    int(N),
        "n_frames":  int(F.shape[1]),
        "runtime_s": float(runtime_s),
    }


# ---------------------------------------------------------------------------
# Run-folder allocation
# ---------------------------------------------------------------------------
def _next_run_id(out_root: Path) -> int:
    out_root.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name.split("_", 1)[0])
        for p in out_root.iterdir()
        if p.is_dir() and p.name[:4].isdigit()
    ]
    return max(existing, default=0) + 1


def _git_rev() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
             "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _flatten(d, prefix="") -> dict:
    """Recursively flatten a nested dict for index columns (a.b.c → a_b_c)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}_{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _append_index(index_csv: Path, row: dict) -> None:
    """Append a row to the runs index, aligning columns to a stable schema.

    Different recipes can have different leaf fields (e.g. the LOWESS branch
    has ``recipe_fluctuations_maxiter`` while percentile does not). We can't
    rely on the row's own key order matching the existing header — that
    silently mis-aligns subsequent rows. Use pandas to do the union-of-columns
    merge correctly.
    """
    import pandas as pd
    new = pd.DataFrame([row])
    if index_csv.exists():
        existing = pd.read_csv(index_csv, dtype={"run_id": str})
        merged = pd.concat([existing, new], ignore_index=True, sort=False)
    else:
        merged = new
    merged.to_csv(index_csv, index=False)


# ---------------------------------------------------------------------------
# Settings (Pydantic CLI)
# ---------------------------------------------------------------------------
from pydantic import Field
from pydantic_settings import BaseSettings


class RunSettings(BaseSettings, cli_parse_args=True):
    """CLI for the baseline_search runner."""

    recipe: Path = Field(
        ..., description="Path to a recipe JSON file."
    )
    inputs_dir: Path = Field(
        Path("/root/capsule/scratch/first_try"),
        description="Directory containing per-session subfolders with F_all_array.npy etc.",
    )
    out: Path = Field(
        Path("/root/capsule/scratch/runs"),
        description="Parent directory for numbered run folders.",
    )
    slug: str = Field(
        ..., description="Short human-readable slug for the run folder name."
    )
    sessions: str = Field(
        ..., description="Comma-separated session keys to fit."
    )
    n_jobs: int = Field(
        -1, description="joblib n_jobs. 1 disables parallelism (helps debug)."
    )
    verbose: int = Field(
        5, description="joblib verbosity level."
    )
    description: str = Field(
        "", description="Optional human description, written to metadata.json."
    )

    class Config:
        env_prefix = "BSEARCH_"


def main() -> int:
    cfg = RunSettings()
    sessions = [s.strip() for s in cfg.sessions.split(",") if s.strip()]

    recipe_text = Path(cfg.recipe).read_text()
    recipe = Recipe.from_json(recipe_text)

    run_id = _next_run_id(cfg.out)
    run_dir = cfg.out / f"{run_id:04d}_{cfg.slug}"
    run_dir.mkdir(parents=True, exist_ok=False)

    # Persist the canonical recipe (re-serialized through Pydantic so it's normalized)
    (run_dir / "recipe.json").write_text(recipe.to_json())

    sess_summaries = []
    for sess in sessions:
        print(f"[run {run_id:04d}] fitting session {sess} ...", flush=True)
        sess_summaries.append(
            _run_session(recipe, cfg.inputs_dir, run_dir, sess, cfg.n_jobs, cfg.verbose)
        )

    metadata = {
        "run_id":      run_id,
        "slug":        cfg.slug,
        "description": cfg.description or recipe.description,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "host":        socket.gethostname(),
        "user":        os.environ.get("USER", "unknown"),
        "git_rev":     _git_rev(),
        "inputs_dir":  str(cfg.inputs_dir.resolve()),
        "recipe_path": str(Path(cfg.recipe).resolve()),
        "sessions":    sess_summaries,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Append a flattened row to the index
    flat_recipe = _flatten(recipe.model_dump())
    index_row = {
        "run_id":      f"{run_id:04d}",
        "slug":        cfg.slug,
        "run_dir":     str(run_dir),
        "created_at":  metadata["created_at"],
        "n_sessions":  len(sessions),
        "description": metadata["description"],
        **{f"recipe_{k}": v for k, v in flat_recipe.items()},
    }
    _append_index(cfg.out / "index.csv", index_row)

    print(f"[run {run_id:04d}] done -> {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
