"""Session data loading — mmap'd arrays, per-session dataclass, aggregate metrics."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PARENT_DIR = Path("/root/capsule/scratch/first_try")
DEFAULT_DATA_DIR   = Path("/root/capsule/data")

METRIC_DISPLAY = {
    "F_noise":          "noise",
    "F_snr":            "snr",
    "bleaching_metric": "bleaching",
    "sustained_metric": "sustained",
    "F_skewness":       "skewness",
}



@dataclass
class SessionData:
    session_key: str
    path: Path
    timestamps: np.ndarray        # (T,)
    F: np.ndarray                 # (N, T)
    baselines: dict               # name -> (N, T): short, long, F0trend, F0
    dffs: dict                    # name -> (N, T): short, long, F0trend, F0
    metrics: pd.DataFrame         # columns = display names, rows aligned with ROI axis
    rois: pd.DataFrame            # plane_id, cell_roi_id, ...

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


def list_sessions(parent_dir: Path = DEFAULT_PARENT_DIR) -> list[Path]:
    """Return subdirs that look like input session folders (contain F_all_array.npy)."""
    return sorted(
        p for p in Path(parent_dir).iterdir()
        if p.is_dir() and (p / "F_all_array.npy").exists()
    )


_BASELINE_FILES = {
    "short":   "baseline_short_window_all_array.npy",
    "long":    "baseline_long_window_all_array.npy",
    "F0trend": "F0trend_all.npy",
    "F0":      "F0_all.npy",
}
_PRECOMP_DFF_FILES = {
    "short": "dff_short_window_all_array.npy",
    "long":  "dff_long_window_all_array.npy",
}


@lru_cache(maxsize=8)
def load_session(session_path_str: str) -> SessionData:
    p = Path(session_path_str)
    if not (p / "F_all_array.npy").exists():
        raise FileNotFoundError(
            f"{p} does not look like a session inputs folder "
            f"(F_all_array.npy not found). "
            f"Did you select a runs folder instead of the inputs folder?"
        )
    F = np.load(p / "F_all_array.npy", mmap_mode="r")
    roi_csv = p / "sczdrift_df_all.csv"
    if roi_csv.exists():
        rois = pd.read_csv(roi_csv)
    else:
        rois = pd.DataFrame({
            "plane_id":    ["unknown"] * F.shape[0],
            "cell_roi_id": list(range(F.shape[0])),
        })
    ts_path = p / "timestamps.npy"
    timestamps = np.load(ts_path) if ts_path.exists() else np.arange(F.shape[1], dtype=np.float32)

    baselines: dict = {}
    dffs: dict = {}
    for key, bfile in _BASELINE_FILES.items():
        bpath = p / bfile
        if not bpath.exists():
            continue
        b = np.load(bpath, mmap_mode="r")
        baselines[key] = b
        dfile = _PRECOMP_DFF_FILES.get(key)
        dpath = p / dfile if dfile else None
        if dpath and dpath.exists():
            dffs[key] = np.load(dpath, mmap_mode="r")
        else:
            dffs[key] = _safe_dff(np.asarray(F), np.asarray(b))

    metrics = pd.DataFrame({
        display: np.load(p / f"{key}.npy")
        for key, display in METRIC_DISPLAY.items()
        if (p / f"{key}.npy").exists()
    })
    return SessionData(session_key=p.name, path=p, timestamps=timestamps,
                       F=F, baselines=baselines, dffs=dffs, metrics=metrics, rois=rois)


@lru_cache(maxsize=1)
def aggregate_metrics(parent_dir_str: str = str(DEFAULT_PARENT_DIR)) -> pd.DataFrame:
    frames = []
    for sess in list_sessions(Path(parent_dir_str)):
        try:
            df = pd.DataFrame({
                display: np.load(sess / f"{key}.npy")
                for key, display in METRIC_DISPLAY.items()
            })
        except FileNotFoundError:
            continue
        df["session_key"] = sess.name
        df["roi_index"] = np.arange(len(df))
        frames.append(df)
    if not frames:
        cols = list(METRIC_DISPLAY.values()) + ["session_key", "roi_index"]
        return pd.DataFrame(columns=cols)
    return pd.concat(frames, ignore_index=True)


_SUBJECT_DATE_RE = re.compile(r"^(?P<subject>\d+)_(?P<date>\d{4}-\d{2}-\d{2})$")


@lru_cache(maxsize=64)
def find_processed_dir(session_key: str,
                        data_dir_str: str = str(DEFAULT_DATA_DIR)) -> Path | None:
    m = _SUBJECT_DATE_RE.match(session_key)
    if not m:
        return None
    pattern = f"multiplane-ophys_{m['subject']}_{m['date']}_*_processed_*"
    matches = sorted(Path(data_dir_str).glob(pattern))
    return matches[-1] if matches else None
