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
    if b_init_from == "half_of_max_F_minus_min_F":
        return (np.max(F_all, axis=1) - np.min(F_all, axis=1)) / 2.0
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
    F_mean = F_all.mean(axis=1)
    N = F_all.shape[0]
    x0 = np.empty((N, 7), dtype=np.float64)
    x0[:, 0] = F_mean
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
def biexp_bright_default_bounds(spec, t_max):
    t_high = t_max * spec.t_high_factor
    return [
        (0.0, None),                          # b_inf
        (0.0, None),                          # b_slow
        (0.0, None),                          # b_fast
        (0.0, None),                          # b_bright
        (spec.t_slow_min, t_high),            # t_slow
        (spec.t_fast_min, spec.t_fast_max),   # t_fast
        (spec.t_bright_min, t_high),          # t_bright
    ]


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
