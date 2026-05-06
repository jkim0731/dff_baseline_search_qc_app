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
    safe = np.abs(b) > 1e-6
    return np.where(safe, (F - b) / np.where(safe, b, 1.0), 0.0).astype(np.float32)


def list_sessions(parent_dir: Path = DEFAULT_PARENT_DIR) -> list[Path]:
    return sorted(p for p in Path(parent_dir).iterdir() if p.is_dir())


@lru_cache(maxsize=8)
def load_session(session_path_str: str) -> SessionData:
    p = Path(session_path_str)
    rois = pd.read_csv(p / "sczdrift_df_all.csv")
    F = np.load(p / "F_all_array.npy", mmap_mode="r")
    timestamps = np.load(p / "timestamps.npy")
    baselines = {
        "short":   np.load(p / "baseline_short_window_all_array.npy", mmap_mode="r"),
        "long":    np.load(p / "baseline_long_window_all_array.npy",  mmap_mode="r"),
        "F0trend": np.load(p / "F0trend_all.npy", mmap_mode="r"),
        "F0":      np.load(p / "F0_all.npy",      mmap_mode="r"),
    }
    dffs = {
        "short":   np.load(p / "dff_short_window_all_array.npy", mmap_mode="r"),
        "long":    np.load(p / "dff_long_window_all_array.npy",  mmap_mode="r"),
        "F0trend": _safe_dff(np.asarray(F), np.asarray(baselines["F0trend"])),
        "F0":      _safe_dff(np.asarray(F), np.asarray(baselines["F0"])),
    }
    metrics = pd.DataFrame({
        display: np.load(p / f"{key}.npy")
        for key, display in METRIC_DISPLAY.items()
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
