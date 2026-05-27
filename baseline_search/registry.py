"""Name → callable maps for the recipe components.

Each `kind` string in a recipe (model, x0, sigma, bounds, M) names an entry
here. To add a new variant: define a function/class, register a new `kind` in
both the `recipe.py` Literal and one of the dicts below.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import jax.numpy as jnp

from baseline_fitting import (
    AsymmetricTukeyBiweight,
    OneSidedTukeyBiweight,
    TukeyBiweight,
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def biexp_bright_v1(params, t, xp=np):
    """7-param model used in long_vs_short_baseline_window.ipynb.

        F(t) = b_inf + b_slow*exp(-t/t_slow)
                     + b_fast*exp(-t/t_fast)
                     - b_bright*exp(-t/t_bright)
    """
    b_inf, b_slow, b_fast, b_bright, t_slow, t_fast, t_bright = params
    E_slow = xp.exp(-t / t_slow)
    E_fast = xp.exp(-t / t_fast)
    E_bright = xp.exp(-t / t_bright)
    return b_inf + b_slow * E_slow + b_fast * E_fast - b_bright * E_bright


MODEL_FNS: dict[str, Callable] = {
    "biexp_bright_v1": biexp_bright_v1,
}


# Each model registers the names of its parameters (for diagnostic columns
# in res_all and metadata) and how many params it expects.
MODEL_PARAM_NAMES: dict[str, list[str]] = {
    "biexp_bright_v1": [
        "b_inf",
        "b_slow",
        "b_fast",
        "b_bright",
        "t_slow",
        "t_fast",
        "t_bright",
    ],
}


# ---------------------------------------------------------------------------
# x0 builders
# ---------------------------------------------------------------------------
def _resolve_b_init(F_all, baseline_long, baseline_short, b_init_from, b_init_value):
    """Per-ROI scalar used as the initial b_slow / b_fast / b_bright."""
    if b_init_from == "mean_F_minus_long_baseline":
        if baseline_long is None:
            raise ValueError(
                "b_init_from='mean_F_minus_long_baseline' requires baseline_long input"
            )
        return np.mean(F_all - baseline_long, axis=1)
    if b_init_from == "mean_F_minus_short_baseline":
        if baseline_short is None:
            raise ValueError(
                "b_init_from='mean_F_minus_short_baseline' requires baseline_short input"
            )
        return np.mean(F_all - baseline_short, axis=1)
    if b_init_from == "zero":
        return np.zeros(F_all.shape[0], dtype=F_all.dtype)
    if b_init_from == "scalar":
        if b_init_value is None:
            raise ValueError("b_init_from='scalar' requires b_init_value")
        return np.full(F_all.shape[0], float(b_init_value), dtype=F_all.dtype)
    raise ValueError(f"Unknown b_init_from: {b_init_from!r}")


def _resolve_t_const(t_max, kind):
    if kind == "t_max/2":
        return t_max / 2.0
    if kind == "t_max/4":
        return t_max / 4.0
    raise ValueError(f"Unknown t-init kind: {kind!r}")


def biexp_bright_default_x0(spec, F_all, t_max, baseline_long, baseline_short):
    """Returns an (N, 7) float array of per-ROI x0 vectors."""
    b_init = _resolve_b_init(
        F_all, baseline_long, baseline_short, spec.b_init_from, spec.b_init_value
    )
    t_slow_init = _resolve_t_const(t_max, spec.t_slow_init_from)
    t_bright_init = _resolve_t_const(t_max, spec.t_bright_init_from)

    b_inf_init_from = getattr(spec, "b_inf_init_from", "mean_F")
    if b_inf_init_from == "last_N_frames":
        n = int(getattr(spec, "b_inf_n_frames", 1000))
        b_inf = F_all[:, -n:].mean(axis=1)
    else:
        b_inf = F_all.mean(axis=1)

    N = F_all.shape[0]
    x0 = np.empty((N, 7), dtype=np.float64)
    x0[:, 0] = b_inf
    x0[:, 1] = b_init
    x0[:, 2] = b_init
    x0[:, 3] = b_init
    x0[:, 4] = t_slow_init
    x0[:, 5] = spec.t_fast_init
    x0[:, 6] = t_bright_init
    return x0


X0_FNS: dict[str, Callable] = {
    "biexp_bright_default": biexp_bright_default_x0,
}


# ---------------------------------------------------------------------------
# Sigma builders
# ---------------------------------------------------------------------------
def noise_std_sigma(spec, F_all):
    """Per-ROI sigma via aind_ophys_utils.signal_utils.noise_std."""
    from aind_ophys_utils.signal_utils import noise_std as _noise_std

    return np.asarray(_noise_std(F_all, spec.method))


def fixed_value_sigma(spec, F_all):
    return np.full(F_all.shape[0], float(spec.value))


def mad_residual_sigma(spec, F_all):
    # Sentinel: returns None per ROI so the runner passes fixed_sigma=None
    # to fit_baseline (forcing it to MAD-estimate inside the IRLS loop).
    return None


SIGMA_FNS: dict[str, Callable] = {
    "noise_std": noise_std_sigma,
    "fixed_value": fixed_value_sigma,
    "mad_residual": mad_residual_sigma,
}


# ---------------------------------------------------------------------------
# Bounds builders
# ---------------------------------------------------------------------------
def biexp_bright_default_bounds(spec, t_max, F_all=None):
    """Return bounds for the 7-param biexp_bright model.

    When spec.b_amp_max_factor is set and F_all (N, T) is provided, returns a
    per-ROI list of length N (each element a list of 7 tuples).  Otherwise
    returns a single shared list of 7 tuples.

    Optional fields on spec:
      b_bright_max_factor: separate ub multiplier for b_bright (defaults to b_amp_max_factor)
      b_inf_lb_factor: b_inf lower bound = factor * P1(F_roi) (defaults to 0)
      t_slow_min_tmax_factor: t_slow lb = max(t_slow_min, factor * t_max)
      t_bright_min_tmax_factor: t_bright lb = max(t_bright_min, factor * t_max)
      b_fast_ptp_window_s: b_fast ub = P99-P1 of first N seconds of trimmed trace
    """
    # Effective time-constant lower bounds (relative or absolute)
    t_slow_min_factor   = getattr(spec, "t_slow_min_tmax_factor",   None)
    t_bright_min_factor = getattr(spec, "t_bright_min_tmax_factor", None)
    t_slow_lb   = max(spec.t_slow_min,   t_slow_min_factor   * t_max) if t_slow_min_factor   is not None else spec.t_slow_min
    t_bright_lb = max(spec.t_bright_min, t_bright_min_factor * t_max) if t_bright_min_factor is not None else spec.t_bright_min

    t_high = t_max * spec.t_high_factor
    time_bounds = [
        (t_slow_lb,              t_high),           # t_slow
        (spec.t_fast_min, spec.t_fast_max),         # t_fast
        (t_bright_lb,            t_high),           # t_bright
    ]
    b_amp_max_factor    = getattr(spec, "b_amp_max_factor",    None)
    b_bright_max_factor = getattr(spec, "b_bright_max_factor", None)
    b_inf_lb_factor     = getattr(spec, "b_inf_lb_factor",     None)
    b_fast_ptp_window_s = getattr(spec, "b_fast_ptp_window_s", None)

    if b_amp_max_factor is not None and F_all is not None:
        # Per-ROI amplitude upper bounds from robust ptp (P99 - P1)
        # F_all is already trimmed (initial seconds removed) by resolve()
        p1  = np.percentile(F_all,  1, axis=1).astype(np.float64)
        p99 = np.percentile(F_all, 99, axis=1).astype(np.float64)
        ptp = p99 - p1
        amp_ubs        = b_amp_max_factor * ptp
        b_bright_factor = b_bright_max_factor if b_bright_max_factor is not None else b_amp_max_factor
        bright_ubs     = b_bright_factor * ptp
        b_inf_lbs      = b_inf_lb_factor * p1 if b_inf_lb_factor is not None else np.zeros(len(p1))

        # b_fast upper bound: ptp of first N seconds of trimmed trace (per ROI)
        if b_fast_ptp_window_s is not None:
            n_win = max(1, min(int(b_fast_ptp_window_s * F_all.shape[1] / t_max), F_all.shape[1]))
            F_win = F_all[:, :n_win]
            fast_ubs = (np.percentile(F_win, 99, axis=1) - np.percentile(F_win, 1, axis=1)).astype(np.float64)
        else:
            fast_ubs = amp_ubs

        return [
            [(float(b_inf_lb), None),
             (0.0, float(amp_ub)),
             (0.0, float(fast_ub)),
             (0.0, float(bright_ub))]
            + time_bounds
            for b_inf_lb, amp_ub, fast_ub, bright_ub in zip(b_inf_lbs, amp_ubs, fast_ubs, bright_ubs)
        ]
    shared_amp = (0.0, None)
    return [(0.0, None), shared_amp, shared_amp, shared_amp] + time_bounds


BOUNDS_FNS: dict[str, Callable] = {
    "biexp_bright_default": biexp_bright_default_bounds,
}


# ---------------------------------------------------------------------------
# M-estimator builders
# ---------------------------------------------------------------------------
def _build_M(spec) -> Any:
    if spec.kind == "AsymmetricTukeyBiweight":
        return AsymmetricTukeyBiweight(c_pos=spec.c_pos, c_neg=spec.c_neg)
    if spec.kind == "OneSidedTukeyBiweight":
        return OneSidedTukeyBiweight(c=spec.c)
    if spec.kind == "TukeyBiweight":
        return TukeyBiweight(c=spec.c)
    raise ValueError(f"Unknown M kind: {spec.kind!r}")


M_FNS: dict[str, Callable] = {
    "AsymmetricTukeyBiweight": _build_M,
    "OneSidedTukeyBiweight": _build_M,
    "TukeyBiweight": _build_M,
}


# ---------------------------------------------------------------------------
# dtype mapping
# ---------------------------------------------------------------------------
DTYPES = {
    "float32": jnp.float32,
    "float64": jnp.float64,
}
