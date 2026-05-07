"""Recipe → concrete fit args.

Reads a Recipe + the per-session input arrays and produces everything needed
to call ``fit_baseline`` for each ROI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .recipe import (
    LowessFluctuations,
    PercentileFluctuations,
    Recipe,
    SameAsTrendMSpec,
)
from .registry import (
    BOUNDS_FNS,
    DTYPES,
    M_FNS,
    MODEL_FNS,
    MODEL_PARAM_NAMES,
    SIGMA_FNS,
    X0_FNS,
    _build_M,
)


@dataclass
class ResolvedFit:
    """Everything needed to call ``fit_baseline`` per ROI in a session."""

    # per-ROI
    F_all: np.ndarray                        # (N, T)
    timestamps: np.ndarray                   # (T,)
    x0_all: np.ndarray                       # (N, n_params)
    sigma_all: Optional[np.ndarray]          # (N,) or None  (None ⇒ MAD inside fit)

    # shared across ROIs
    model_fn: Callable
    model_param_names: list[str]
    bounds: list[tuple]
    M: Any
    M_fluctuations: Optional[Any]            # None ⇒ fit_baseline default (M.with_xp(np))
    fit_baseline_kwargs: dict


def _resolve_M_fluctuations(rec: Recipe, trend_M: Any) -> Optional[Any]:
    fluct = rec.fluctuations
    if isinstance(fluct, PercentileFluctuations):
        return None  # M is not used in the percentile branch
    # LOWESS branch
    if isinstance(fluct.M, SameAsTrendMSpec):
        return None  # let fit_baseline default to trend_M.with_xp(np)
    return _build_M(fluct.M)


def _fluctuations_kwargs(rec: Recipe) -> dict:
    """Map the FluctuationsSpec union into fit_baseline kwargs."""
    fluct = rec.fluctuations
    if isinstance(fluct, LowessFluctuations):
        return dict(
            method="lowess",
            mode=fluct.mode,
            frac=fluct.frac,
            maxiter=fluct.maxiter,
            tol=fluct.tol,
            percentile=None,
        )
    if isinstance(fluct, PercentileFluctuations):
        return dict(
            method="percentile",
            mode=fluct.mode,
            frac=fluct.frac,
            percentile=fluct.percentile,
            maxiter=rec.fit.maxiter,  # ignored by percentile branch
            tol=rec.fit.tol,           # ignored by percentile branch
        )
    raise TypeError(f"Unknown FluctuationsSpec: {type(fluct).__name__}")


def resolve(
    recipe: Recipe,
    F_all: np.ndarray,
    timestamps: np.ndarray,
    *,
    baseline_long: Optional[np.ndarray] = None,
    baseline_short: Optional[np.ndarray] = None,
) -> ResolvedFit:
    if F_all.ndim != 2:
        raise ValueError(f"F_all must be (N, T); got shape {F_all.shape}")
    if timestamps.ndim != 1 or timestamps.shape[0] != F_all.shape[1]:
        raise ValueError(
            f"timestamps must be (T,) matching F_all.shape[1]={F_all.shape[1]}; "
            f"got {timestamps.shape}"
        )

    t_max = float(timestamps[-1])

    model_fn = MODEL_FNS[recipe.model.kind]
    param_names = MODEL_PARAM_NAMES[recipe.model.kind]

    x0_all = X0_FNS[recipe.x0.kind](
        recipe.x0,
        F_all,
        t_max=t_max,
        baseline_long=baseline_long,
        baseline_short=baseline_short,
    )
    if x0_all.shape != (F_all.shape[0], len(param_names)):
        raise ValueError(
            f"x0 builder produced shape {x0_all.shape}, "
            f"expected {(F_all.shape[0], len(param_names))}"
        )

    sigma_all = SIGMA_FNS[recipe.sigma.kind](recipe.sigma, F_all)
    if sigma_all is not None and sigma_all.shape != (F_all.shape[0],):
        raise ValueError(
            f"sigma builder produced shape {sigma_all.shape}, "
            f"expected {(F_all.shape[0],)}"
        )

    bounds = BOUNDS_FNS[recipe.bounds.kind](recipe.bounds, t_max=t_max)
    if len(bounds) != len(param_names):
        raise ValueError(
            f"bounds builder produced {len(bounds)} entries, "
            f"expected {len(param_names)}"
        )

    M = _build_M(recipe.M)
    M_fluctuations = _resolve_M_fluctuations(recipe, M)

    fit_kwargs: dict = dict(
        backend=recipe.fit.backend,
        dtype=DTYPES[recipe.fit.dtype],
        maxiter=recipe.fit.maxiter,
        tol=recipe.fit.tol,
        optimizer_options=recipe.fit.optimizer_options.model_dump(),
    )
    fit_kwargs.update(_fluctuations_kwargs(recipe))

    return ResolvedFit(
        F_all=F_all,
        timestamps=timestamps,
        x0_all=x0_all,
        sigma_all=sigma_all,
        model_fn=model_fn,
        model_param_names=param_names,
        bounds=bounds,
        M=M,
        M_fluctuations=M_fluctuations,
        fit_baseline_kwargs=fit_kwargs,
    )
