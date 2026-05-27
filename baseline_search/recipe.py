"""Pydantic models for a baseline-fitting recipe.

A recipe encodes the *protocol* that derives every input to ``fit_baseline``
(model, x0, sigma, bounds, M, fluctuations, fit options) — not just resolved
scalar values. This is what gets serialized to ``recipe.json`` per run.

Discriminated unions (on ``kind`` for components, ``method`` for fluctuations)
mean every recipe has exactly one valid shape per component, and Pydantic
rejects unknown fields and missing required fields at load time.
"""

from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared base — forbid unknown fields so typos in JSON fail loudly.
# ---------------------------------------------------------------------------
class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Model (parametric trend)
# ---------------------------------------------------------------------------
class BiexpBrightV1ModelSpec(_StrictModel):
    """7-param model: b_inf + b_slow*E_slow + b_fast*E_fast - b_bright*E_bright."""

    kind: Literal["biexp_bright_v1"] = "biexp_bright_v1"


ModelSpec = Annotated[
    Union[BiexpBrightV1ModelSpec],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# x0 (initial parameter vector for the trend fit)
# ---------------------------------------------------------------------------
class BiexpBrightDefaultX0(_StrictModel):
    """Initial x0 for the 7-param biexp_bright_v1 model.

    b_inf is initialised from the last ``b_inf_n_frames`` frames of the
    (trimmed) trace when ``b_inf_init_from="last_N_frames"``; otherwise
    from the whole-trace mean.
    """

    kind: Literal["biexp_bright_default"] = "biexp_bright_default"
    b_init_from: Literal[
        "mean_F_minus_long_baseline",
        "mean_F_minus_short_baseline",
        "zero",
        "scalar",
    ] = "mean_F_minus_long_baseline"
    b_init_value: Optional[float] = None  # required iff b_init_from == "scalar"
    t_fast_init: float = 60.0
    t_slow_init_from: Literal["t_max/2", "t_max/4"] = "t_max/2"
    t_bright_init_from: Literal["t_max/2", "t_max/4"] = "t_max/2"
    b_inf_init_from: Literal["mean_F", "last_N_frames"] = "last_N_frames"
    b_inf_n_frames: int = 1000


X0Spec = Annotated[
    Union[BiexpBrightDefaultX0],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Sigma (per-ROI fixed scale for the M-estimator)
# ---------------------------------------------------------------------------
class NoiseStdSigma(_StrictModel):
    """``aind_ophys_utils.signal_utils.noise_std(F_all_array, method)``."""

    kind: Literal["noise_std"] = "noise_std"
    method: Literal["mad", "fft", "welch"] = "mad"


class FixedValueSigma(_StrictModel):
    """A single scalar applied to every ROI."""

    kind: Literal["fixed_value"] = "fixed_value"
    value: float


class MadResidualSigma(_StrictModel):
    """Skip ``fixed_sigma`` — let ``nonlinear_fit`` recompute MAD per IRLS iter."""

    kind: Literal["mad_residual"] = "mad_residual"


SigmaSpec = Annotated[
    Union[NoiseStdSigma, FixedValueSigma, MadResidualSigma],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------
class BiexpBrightDefaultBounds(_StrictModel):
    """Bounds for the 7-param biexp_bright model.

    When ``b_amp_max_factor`` is set, the upper bound for b_slow and b_fast is
    ``b_amp_max_factor * (P99-P1)(F_roi)`` per ROI.  Set to None for unbounded.

    ``b_bright_max_factor``: separate ub for b_bright; defaults to b_amp_max_factor.
    ``b_inf_lb_factor``: lower bound for b_inf = factor * P1(F_roi). Default None (lb=0).

    Relative time-constant lower bounds (effective lb = max(absolute_min, factor * t_max)):
    ``t_slow_min_tmax_factor``: sets t_slow lb = max(t_slow_min, factor * t_max).
    ``t_bright_min_tmax_factor``: sets t_bright lb = max(t_bright_min, factor * t_max).

    ``b_fast_ptp_window_s``: if set, b_fast ub = P99-P1 of the first N seconds of the
    trimmed trace (after skip_initial_seconds), rather than b_amp_max_factor * global ptp.
    """

    kind: Literal["biexp_bright_default"] = "biexp_bright_default"
    t_high_factor: float = 5.0
    t_fast_max: float = 300.0
    t_slow_min: float = 300.0
    t_bright_min: float = 300.0
    t_fast_min: float = 30.0
    b_amp_max_factor: Optional[float] = 2.0
    b_bright_max_factor: Optional[float] = None  # if None, falls back to b_amp_max_factor
    b_inf_lb_factor: Optional[float] = None      # if None, b_inf lb = 0
    t_slow_min_tmax_factor: Optional[float] = None    # if None, uses t_slow_min only
    t_bright_min_tmax_factor: Optional[float] = None  # if None, uses t_bright_min only
    b_fast_ptp_window_s: Optional[float] = None       # if None, uses b_amp_max_factor * global ptp


BoundsSpec = Annotated[
    Union[BiexpBrightDefaultBounds],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# M-estimator (RobustNorm)
# ---------------------------------------------------------------------------
class AsymTukeyMSpec(_StrictModel):
    kind: Literal["AsymmetricTukeyBiweight"] = "AsymmetricTukeyBiweight"
    c_pos: float
    c_neg: float


class OneSidedTukeyMSpec(_StrictModel):
    kind: Literal["OneSidedTukeyBiweight"] = "OneSidedTukeyBiweight"
    c: float


class TukeyMSpec(_StrictModel):
    kind: Literal["TukeyBiweight"] = "TukeyBiweight"
    c: float


MSpec = Annotated[
    Union[AsymTukeyMSpec, OneSidedTukeyMSpec, TukeyMSpec],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Fluctuations (LOWESS or percentile branch)
# ---------------------------------------------------------------------------
class SameAsTrendMSpec(_StrictModel):
    """Sentinel: reuse the trend-stage M-estimator for LOWESS IRLS."""

    kind: Literal["same_as_trend"] = "same_as_trend"


FluctuationsMSpec = Annotated[
    Union[AsymTukeyMSpec, OneSidedTukeyMSpec, TukeyMSpec, SameAsTrendMSpec],
    Field(discriminator="kind"),
]


class LowessFluctuations(_StrictModel):
    method: Literal["lowess"] = "lowess"
    mode: Literal["ratio", "subtract"] = "ratio"
    frac: float = Field(default=0.1, gt=0.0, le=1.0)
    M: FluctuationsMSpec = Field(default_factory=lambda: SameAsTrendMSpec())
    maxiter: int = 5
    tol: float = 1e-3


class PercentileFluctuations(_StrictModel):
    method: Literal["percentile"] = "percentile"
    mode: Literal["ratio", "subtract"] = "ratio"
    frac: float = Field(default=0.1, gt=0.0, le=1.0)
    percentile: Optional[float] = Field(default=None, ge=0.0, le=100.0)


FluctuationsSpec = Annotated[
    Union[LowessFluctuations, PercentileFluctuations],
    Field(discriminator="method"),
]


# ---------------------------------------------------------------------------
# Fit (optimizer / backend options for the trend stage)
# ---------------------------------------------------------------------------
class OptimizerOptions(_StrictModel):
    maxiter: int = 20000
    ftol: float = 1e-12
    gtol: float = 1e-10


class FitSpec(_StrictModel):
    backend: Literal["numpy", "jax"] = "jax"
    dtype: Literal["float32", "float64"] = "float32"
    maxiter: int = 5
    tol: float = 1e-3
    optimizer_options: OptimizerOptions = Field(default_factory=OptimizerOptions)


# ---------------------------------------------------------------------------
# Top-level Recipe
# ---------------------------------------------------------------------------
class Recipe(_StrictModel):
    schema_version: Literal[1] = 1
    description: str = ""

    model: ModelSpec
    x0: X0Spec
    sigma: SigmaSpec
    bounds: BoundsSpec
    M: MSpec
    fluctuations: FluctuationsSpec
    fit: FitSpec = Field(default_factory=FitSpec)
    skip_initial_seconds: float = 5.0

    def to_json(self, *, indent: int = 2) -> str:
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "Recipe":
        return cls.model_validate_json(text)
