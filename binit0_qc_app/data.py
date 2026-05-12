"""Session + run data loading for the binit0 noise-criterion QC app."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# ── default paths ─────────────────────────────────────────────────────────────
INPUTS_755  = Path("/results/runs/0000_first_try")
INPUTS_804  = Path("/results/runs/804670_inputs")
RUNS_DIR    = Path("/results/runs")
TARGET_COEF = 0.674   # half-normal median: E[|X| | X<0] for X~N(0,σ²) = 0.6745σ

# ── combo definitions ─────────────────────────────────────────────────────────
COMBOS: list[tuple[int, int]] = [(2,3),(2,4),(2,5),(3,3),(3,4),(3,5),(4,5)]

COMBO_KEY   = {c: f"c{''.join(map(str,c))}" for c in COMBOS}   # (2,3)->'c23'
COMBO_LABEL = {c: f"({c[0]},{c[1]})" for c in COMBOS}          # (2,3)->'(2,3)'
KEY_COMBO   = {v: k for k, v in COMBO_KEY.items()}              # 'c23'->(2,3)
COMBO_KEY_LIST = [COMBO_KEY[c] for c in COMBOS]                 # ordered keys

_COMBO_RUN_NAMES = {
    (2,3): "0017_cpos2_cneg3_lowess_binit0",
    (2,4): "0018_cpos2_cneg4_lowess_binit0",
    (2,5): "0020_cpos2_cneg5_lowess_binit0",
    (3,3): "0019_cpos3_cneg3_lowess_binit0",
    (3,4): "0021_cpos3_cneg4_lowess_binit0",
    (3,5): "0022_cpos3_cneg5_lowess_binit0",
    (4,5): "0024_cpos4_cneg5_lowess_binit0",
}
COMBO_RUN = {c: RUNS_DIR / name for c, name in _COMBO_RUN_NAMES.items()}

TRACE_KEYS = ["short", "long"] + COMBO_KEY_LIST   # 9 total
COMBO_KEYS = COMBO_KEY_LIST                        # 7 combo-only keys

METRIC_DISPLAY = {
    "F_noise":          "noise",
    "F_snr":            "snr",
    "bleaching_metric": "bleaching",
    "sustained_metric": "sustained",
    "F_skewness":       "skewness",
}


# ── data container ────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    session_key: str
    inputs_dir:  Path
    timestamps:  np.ndarray    # (T,)
    F:           np.ndarray    # (N, T) mmap
    noise:       np.ndarray    # (N,) per-roi noise_std(F, 'mad')
    baselines:   dict          # key->(N,T) mmap: 'short','long','c23',...
    dff_short:   np.ndarray    # (N, T) mmap, precomputed
    dff_long:    np.ndarray    # (N, T) mmap, precomputed
    f0_arrays:   dict          # combo_key->(N,T) mmap F0 (for residuals)
    metrics:     pd.DataFrame  # per-roi, aligned with ROI axis
    rois:        pd.DataFrame  # plane_id, cell_roi_id, ...

    @property
    def n_rois(self) -> int:
        return self.F.shape[0]


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_dff(F: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float32)
    b = np.asarray(baseline, dtype=np.float32)
    nan_mask = np.isnan(b)
    safe = (~nan_mask) & (np.abs(b) > 1e-6)
    result = np.where(safe, (F - b) / np.where(safe, b, 1.0), 0.0).astype(np.float32)
    result[nan_mask] = np.nan
    return result


def _inputs_dir_for(session_key: str) -> Path:
    return INPUTS_804 if session_key.startswith("804670") else INPUTS_755


# ── session discovery ─────────────────────────────────────────────────────────

def list_sessions(
    inputs_dirs: list[Path] | None = None,
    runs_dir:    Path | None       = None,
) -> list[tuple[str, Path]]:
    """[(session_key, inputs_dir)] for sessions with all 7 combo run outputs."""
    if inputs_dirs is None:
        inputs_dirs = [d for d in (INPUTS_755, INPUTS_804) if d.exists()]
    combo_run = {c: (runs_dir or RUNS_DIR) / _COMBO_RUN_NAMES[c] for c in COMBOS}
    sessions: list[tuple[str, Path]] = []
    for inp_dir in inputs_dirs:
        for p in sorted(inp_dir.iterdir()):
            if not (p.is_dir() and (p / "F_all_array.npy").exists()):
                continue
            if all((combo_run[c] / p.name / "F0_all.npy").exists() for c in COMBOS):
                sessions.append((p.name, inp_dir))
    return sessions


# ── session loading ───────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def load_session(
    session_key:    str,
    inputs_dir_str: str,
    runs_dir_str:   str = str(RUNS_DIR),
) -> SessionData:
    inp = Path(inputs_dir_str) / session_key
    rd  = Path(runs_dir_str)

    F     = np.load(inp / "F_all_array.npy",  mmap_mode="r")
    ts    = np.load(inp / "timestamps.npy")
    noise = np.load(inp / "F_noise.npy")

    baselines: dict = {
        "short": np.load(inp / "baseline_short_window_all_array.npy", mmap_mode="r"),
        "long":  np.load(inp / "baseline_long_window_all_array.npy",  mmap_mode="r"),
    }
    dff_short = np.load(inp / "dff_short_window_all_array.npy", mmap_mode="r")
    dff_long  = np.load(inp / "dff_long_window_all_array.npy",  mmap_mode="r")

    f0_arrays: dict = {}
    combo_run_local = {c: rd / _COMBO_RUN_NAMES[c] for c in COMBOS}
    for combo in COMBOS:
        key      = COMBO_KEY[combo]
        sess_run = combo_run_local[combo] / session_key
        baselines[key] = np.load(sess_run / "F0trend_all.npy", mmap_mode="r")
        f0_arrays[key] = np.load(sess_run / "F0_all.npy",      mmap_mode="r")

    rois_csv = inp / "sczdrift_df_all.csv"
    rois = (pd.read_csv(rois_csv) if rois_csv.exists() else
            pd.DataFrame({"plane_id": ["unknown"] * F.shape[0],
                          "cell_roi_id": list(range(F.shape[0]))}))
    metrics = pd.DataFrame({
        disp: np.load(inp / f"{key}.npy")
        for key, disp in METRIC_DISPLAY.items()
        if (inp / f"{key}.npy").exists()
    })
    return SessionData(
        session_key=session_key,
        inputs_dir=inp,
        timestamps=ts,
        F=F, noise=noise,
        baselines=baselines,
        dff_short=dff_short, dff_long=dff_long,
        f0_arrays=f0_arrays,
        metrics=metrics, rois=rois,
    )


# ── noise-criterion computation ───────────────────────────────────────────────

def compute_noise_bar(
    roi_idx: int, sd: SessionData
) -> tuple[dict, float, str | None]:
    """Compute |median(neg residuals)| per combo, target, and winner key.

    Returns (med_neg: dict[key->float], target: float, winner_key: str|None).
    """
    F_roi  = np.asarray(sd.F[roi_idx], dtype=np.float64)
    target = TARGET_COEF * float(sd.noise[roi_idx])
    med_neg: dict = {}
    for key in COMBO_KEYS:
        f0    = np.asarray(sd.f0_arrays[key][roi_idx], dtype=np.float64)
        valid = ~np.isnan(f0)
        res   = F_roi - f0
        neg   = res[valid & (res < 0)]
        med_neg[key] = float(np.median(np.abs(neg))) if len(neg) > 10 else float("nan")

    winner_key, best_dist = None, float("inf")
    for key, val in med_neg.items():
        if np.isfinite(val) and abs(val - target) < best_dist:
            best_dist  = abs(val - target)
            winner_key = key
    return med_neg, target, winner_key


# ── aggregate metrics ─────────────────────────────────────────────────────────

def aggregate_metrics(sessions: list[tuple[str, Path]]) -> pd.DataFrame:
    frames = []
    for sess_key, inp_dir in sessions:
        inp = Path(inp_dir) / sess_key
        try:
            df = pd.DataFrame({
                disp: np.load(inp / f"{key}.npy")
                for key, disp in METRIC_DISPLAY.items()
                if (inp / f"{key}.npy").exists()
            })
        except Exception:
            continue
        df["session_key"] = sess_key
        df["roi_index"]   = np.arange(len(df))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(METRIC_DISPLAY.values()) + ["session_key", "roi_index"])
    return pd.concat(frames, ignore_index=True)
