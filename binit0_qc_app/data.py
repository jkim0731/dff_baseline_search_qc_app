"""Session + run data loading for the binit0 noise-criterion QC app.

All paths are discovered at runtime from a user-supplied runs_dir; there are
no hardcoded filesystem paths in this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COEF = 0.674   # half-normal median: E[|X| | X<0] for X~N(0,σ²) = 0.6745σ

# ── combo definitions (parameter values only, no paths) ───────────────────────
COMBOS: list[tuple[int, int]] = [(2,3),(2,4),(2,5),(3,3),(3,4),(3,5),(4,4),(4,5)]

COMBO_KEY      = {c: f"c{''.join(map(str,c))}" for c in COMBOS}  # (2,3) -> 'c23'
COMBO_LABEL    = {c: f"({c[0]},{c[1]})" for c in COMBOS}         # (2,3) -> '(2,3)'
KEY_COMBO      = {v: k for k, v in COMBO_KEY.items()}             # 'c23' -> (2,3)
COMBO_KEY_LIST = [COMBO_KEY[c] for c in COMBOS]                   # ordered

TRACE_KEYS = ["short", "long"] + COMBO_KEY_LIST   # 10 total
COMBO_KEYS = COMBO_KEY_LIST                        # 8 combo-only keys

METRIC_DISPLAY = {
    "F_noise":          "noise",
    "F_snr":            "snr",
    "bleaching_metric": "bleaching",
    "sustained_metric": "sustained",
    "F_skewness":       "skewness",
}


# ── run discovery ─────────────────────────────────────────────────────────────

def discover_combo_runs(runs_dir: Path) -> dict[tuple[int, int], Path]:
    """Scan runs_dir for folders whose recipe.json matches a binit0 combo.

    A folder qualifies when:
      - recipe.json exists
      - x0.b_init_from == "zero"
      - M.kind == "AsymmetricTukeyBiweight"
      - (M.c_pos, M.c_neg) is one of the 8 COMBOS

    Returns {(c_pos, c_neg): run_dir} for every found combo.
    """
    found: dict[tuple[int, int], Path] = {}
    for d in sorted(runs_dir.iterdir()):
        rp = d / "recipe.json"
        if not d.is_dir() or not rp.exists():
            continue
        try:
            recipe = json.loads(rp.read_text())
        except Exception:
            continue
        x0 = recipe.get("x0", {})
        M  = recipe.get("M", {})
        if (x0.get("b_init_from") == "zero"
                and M.get("kind") == "AsymmetricTukeyBiweight"):
            combo = (int(M["c_pos"]), int(M["c_neg"]))
            if combo in COMBOS and combo not in found:
                found[combo] = d
    return found


def discover_input_dirs(runs_dir: Path, combo_runs: dict[tuple[int, int], Path]) -> list[Path]:
    """Find input directories inside runs_dir by content, not by stored paths.

    An input directory is any subdir of runs_dir that:
    - is NOT a combo run folder (no recipe.json at its root)
    - contains at least one session subdir with F_all_array.npy
    """
    combo_run_paths = {str(p) for p in combo_runs.values()}
    dirs: list[Path] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or str(d) in combo_run_paths:
            continue
        if (d / "recipe.json").exists():
            continue
        if any((s / "F_all_array.npy").exists() for s in d.iterdir() if s.is_dir()):
            dirs.append(d)
    return dirs


# ── data container ────────────────────────────────────────────────────────────

@dataclass
class SessionData:
    session_key: str
    inputs_dir:  Path
    timestamps:  np.ndarray    # (T,)
    F:           np.ndarray    # (N, T) mmap
    noise:       np.ndarray    # (N,) per-roi noise_std(F, 'mad')
    baselines:   dict          # key -> (N,T) mmap: 'short','long','c23',...
    dff_short:   np.ndarray    # (N, T) mmap precomputed
    dff_long:    np.ndarray    # (N, T) mmap precomputed
    f0_arrays:   dict          # combo_key -> (N,T) mmap F0 (for residuals)
    metrics:     pd.DataFrame  # per-roi, aligned with ROI axis
    rois:        pd.DataFrame  # plane_id, cell_roi_id, ...

    @property
    def n_rois(self) -> int:
        return self.F.shape[0]


def _safe_dff(F: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float32)
    b = np.asarray(baseline, dtype=np.float32)
    nan_mask = np.isnan(b)
    safe = (~nan_mask) & (np.abs(b) > 1e-6)
    result = np.where(safe, (F - b) / np.where(safe, b, 1.0), 0.0).astype(np.float32)
    result[nan_mask] = np.nan
    return result


# ── session listing ───────────────────────────────────────────────────────────

def list_sessions(
    runs_dir: Path,
    combo_runs: dict[tuple[int, int], Path] | None = None,
) -> list[tuple[str, Path]]:
    """Return [(session_key, inputs_dir)] for sessions present in all 8 combo runs.

    Raises RuntimeError if any of the 7 required combos is missing from runs_dir.
    """
    if combo_runs is None:
        combo_runs = discover_combo_runs(runs_dir)

    missing = [c for c in COMBOS if c not in combo_runs]
    if missing:
        raise RuntimeError(
            f"Could not find binit0 run folders for combos {missing} in '{runs_dir}'.\n"
            "Each run folder needs recipe.json with x0.b_init_from='zero' and "
            "M.kind='AsymmetricTukeyBiweight'."
        )

    input_dirs = discover_input_dirs(runs_dir, combo_runs)
    sessions: list[tuple[str, Path]] = []
    for inp_dir in input_dirs:
        for p in sorted(inp_dir.iterdir()):
            if not (p.is_dir() and (p / "F_all_array.npy").exists()):
                continue
            if all(
                (combo_runs[c] / p.name / "F0_all.npy").exists()
                for c in COMBOS
            ):
                sessions.append((p.name, inp_dir))
    return sessions


# ── session loading ───────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def load_session(
    session_key: str,
    inputs_dir_str: str,
    combo_run_strs: tuple[str, ...],   # run_dir strings in COMBOS order (hashable)
) -> SessionData:
    inp       = Path(inputs_dir_str) / session_key
    combo_run = {c: Path(s) for c, s in zip(COMBOS, combo_run_strs)}

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
    for combo in COMBOS:
        key      = COMBO_KEY[combo]
        sess_run = combo_run[combo] / session_key
        baselines[key] = np.load(sess_run / "F0trend_all.npy", mmap_mode="r")
        f0_arrays[key] = np.load(sess_run / "F0_all.npy",      mmap_mode="r")

    rois_csv = inp / "sczdrift_df_all.csv"
    rois = (pd.read_csv(rois_csv) if rois_csv.exists() else
            pd.DataFrame({"plane_id":    ["unknown"] * F.shape[0],
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
    roi_idx: int, sd: SessionData, use_f0trend: bool = False,
) -> tuple[dict, float, str | None]:
    """Compute |median(neg residuals)| per combo, the target, and winner key.

    use_f0trend=True  → residuals from F0trend (IRLS trend only)
    use_f0trend=False → residuals from F0     (full LOWESS baseline)
    """
    F_roi  = np.asarray(sd.F[roi_idx], dtype=np.float64)
    target = TARGET_COEF * float(sd.noise[roi_idx])
    med_neg: dict = {}
    for key in COMBO_KEYS:
        src   = sd.baselines[key] if use_f0trend else sd.f0_arrays[key]
        f0    = np.asarray(src[roi_idx], dtype=np.float64)
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
    cols = list(METRIC_DISPLAY.values()) + ["session_key", "roi_index"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)
