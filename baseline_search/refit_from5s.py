"""Re-fit all existing runs for one session, dropping frames where t < T_START.

Outputs are NaN-padded back to the original array length so the QC app can
display them on the same time axis as the full F trace (which starts at t=0).

Usage:
    python -m baseline_search.refit_from5s \
        --session 755252_2024-11-19 \
        --runs_dir /results/runs \
        --inputs_dir /results/runs/first_try \
        --t_start 5.0
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import baseline_search  # noqa: F401
    from baseline_search.recipe import Recipe
    from baseline_search.resolve import resolve
else:
    from .recipe import Recipe
    from .resolve import resolve

from baseline_fitting import fit_baseline  # noqa: E402


def _fit_one_roi(F_row, timestamps, model_fn, x0, bounds, M, M_fluctuations,
                 sigma, fit_baseline_kwargs):
    F0, F0trend, res, info = fit_baseline(
        F_row, timestamps, model_fn, x0, bounds=bounds, M=M,
        M_fluctuations=M_fluctuations, fixed_sigma=sigma, **fit_baseline_kwargs,
    )
    method = fit_baseline_kwargs.get("method", "lowess")
    mode   = fit_baseline_kwargs.get("mode",   "ratio")
    if method == "lowess":
        ls_sigma = info["lowess_sigma"]
        ls = ls_sigma * F0trend if mode == "ratio" else ls_sigma * np.ones_like(F0trend)
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
                else {"percentile": float(info["percentile"]), "size": int(info["size"])}
            ),
        },
    }
    return np.asarray(F0trend), np.asarray(F0), np.asarray(res.x, dtype=np.float64), loss, info_summary


def refit_run(run_dir: Path, session: str, inputs_dir: Path,
              t_start: float, n_jobs: int = -1, verbose: int = 5) -> None:
    sess_in  = inputs_dir / session
    sess_out = run_dir / session

    F_full  = np.load(sess_in / "F_all_array.npy")
    ts_full = np.load(sess_in / "timestamps.npy")

    cut = int(np.searchsorted(ts_full, t_start, side="left"))
    if cut == 0:
        print(f"  [{run_dir.name}] no frames before {t_start}s — skipping cut")
    else:
        print(f"  [{run_dir.name}] dropping first {cut} frames (t < {t_start}s)")

    ts   = ts_full[cut:]
    F    = F_full[:, cut:]
    bl_long_path  = sess_in / "baseline_long_window_all_array.npy"
    bl_short_path = sess_in / "baseline_short_window_all_array.npy"
    bl_long  = np.load(bl_long_path)[:, cut:]  if bl_long_path.exists()  else None
    bl_short = np.load(bl_short_path)[:, cut:] if bl_short_path.exists() else None

    recipe = Recipe.from_json((run_dir / "recipe.json").read_text())
    rf = resolve(recipe, F, ts, baseline_long=bl_long, baseline_short=bl_short)

    N = F.shape[0]
    sigma_iter = [None] * N if rf.sigma_all is None else [float(s) for s in rf.sigma_all]

    def _job(i):
        return _fit_one_roi(
            F[i], rf.timestamps, rf.model_fn, rf.x0_all[i].tolist(),
            rf.bounds, rf.M, rf.M_fluctuations, sigma_iter[i],
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
    print(f"  [{run_dir.name}] fit done in {runtime_s:.1f}s")

    T_full = F_full.shape[1]

    # NaN-pad back to full length
    F0trend_full = np.full((N, T_full), np.nan, dtype=np.float32)
    F0_full      = np.full((N, T_full), np.nan, dtype=np.float64)
    params_all   = np.stack([r[2] for r in results]).astype(np.float64)
    loss_all     = np.asarray([r[3] for r in results], dtype=np.float64)
    info_per_roi = [r[4] for r in results]

    for i, (f0trend, f0, *_) in enumerate(results):
        F0trend_full[i, cut:] = f0trend.astype(np.float32)
        F0_full[i, cut:]      = f0.astype(np.float64)

    sess_out.mkdir(parents=True, exist_ok=True)
    np.save(sess_out / "F0trend_all.npy", F0trend_full)
    np.save(sess_out / "F0_all.npy",      F0_full)
    np.save(sess_out / "res_all.npy",     params_all)
    np.save(sess_out / "loss_all.npy",    loss_all)
    with open(sess_out / "info.json", "w") as fh:
        json.dump({"param_names": rf.model_param_names, "per_roi": info_per_roi},
                  fh, indent=2)
    print(f"  [{run_dir.name}] saved to {sess_out}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--session",    required=True,
                   help="Comma-separated session keys to fit.")
    p.add_argument("--runs_dir",   default="/results/runs")
    p.add_argument("--inputs_dir", default="/results/runs/0000_first_try")
    p.add_argument("--t_start",    type=float, default=5.0)
    p.add_argument("--n_jobs",     type=int,   default=-1)
    p.add_argument("--verbose",    type=int,   default=5)
    p.add_argument("--run_dirs",   nargs="*",  default=None,
                   help="Specific run dir names to process (default: all numbered dirs)")
    args = p.parse_args()

    sessions   = [s.strip() for s in args.session.split(",") if s.strip()]
    runs_dir   = Path(args.runs_dir)
    inputs_dir = Path(args.inputs_dir)

    if args.run_dirs:
        run_dirs = [runs_dir / d for d in args.run_dirs]
    else:
        run_dirs = sorted(
            p for p in runs_dir.iterdir()
            if p.is_dir() and p.name[:4].isdigit()
        )

    print(f"Refitting {len(run_dirs)} run(s), {len(sessions)} session(s) "
          f"(t >= {args.t_start}s) ...")
    for rd in run_dirs:
        if not (rd / "recipe.json").exists():
            print(f"  Skipping {rd.name}: no recipe.json")
            continue
        for sess in sessions:
            refit_run(rd, sess, inputs_dir,
                      t_start=args.t_start, n_jobs=args.n_jobs, verbose=args.verbose)

    print("All done.")


if __name__ == "__main__":
    main()
