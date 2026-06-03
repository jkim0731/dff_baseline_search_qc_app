"""Data loading from production pipeline output (h5 files)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np

DATA_DIR = Path("/root/capsule/data")


def list_session_dirs(data_dir: Path = DATA_DIR) -> list[Path]:
    return sorted(p for p in data_dir.iterdir() if p.is_dir())


def list_planes(session_dir: Path) -> list[str]:
    return sorted(
        p.name for p in session_dir.iterdir()
        if p.is_dir() and p.name.startswith("VISp_")
    )


def _get_fs(session_dir: Path, plane_id: str) -> float:
    proc = session_dir / plane_id / "dff" / f"{plane_id}_df_f_data_process.json"
    if proc.exists():
        try:
            d = json.loads(proc.read_text())
            return float(d["parameters"]["triexp_config"]["fs"])
        except Exception:
            pass
    return 10.71


@dataclass
class PlaneData:
    session_dir: Path
    plane_id: str
    fs: float
    n_frames: int
    n_rois: int
    corrected_F: np.ndarray    # (N, T) float32 — neuropil-corrected F
    baseline: np.ndarray       # (N, T) float32 — F0 in absolute units
    dff: np.ndarray            # (N, T) float32 — (F−F0)/F0 pre-computed
    events: np.ndarray         # (N, T) float32 — OASIS events
    soma_pred: np.ndarray      # (N,) int64
    dendrite_pred: np.ndarray  # (N,) int64
    border: np.ndarray         # (N,) bool
    max_img: np.ndarray        # (H, W) float32
    mean_img: np.ndarray       # (H, W) float32
    roi_coords: np.ndarray     # (3, n_px): [roi_idx, y, x]
    img_shape: tuple[int, int]

    @property
    def timestamps(self) -> np.ndarray:
        return np.arange(self.n_frames, dtype=np.float32) / self.fs

    def is_valid(self, roi_idx: int) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.soma_pred[roi_idx] != 1:
            reasons.append("Not classified as soma")
        if self.dendrite_pred[roi_idx] == 1:
            reasons.append("Classified as dendrite")
        if bool(self.border[roi_idx]):
            reasons.append("Border ROI")
        return len(reasons) == 0, reasons

    def has_data(self, roi_idx: int) -> bool:
        return bool(np.isfinite(self.baseline[roi_idx, 0]))


@lru_cache(maxsize=64)
def load_plane(session_dir_str: str, plane_id: str) -> PlaneData:
    session_dir = Path(session_dir_str)
    plane_dir = session_dir / plane_id

    with h5py.File(plane_dir / "extraction" / f"{plane_id}_extraction.h5", "r") as f:
        corrected_F = np.array(f["traces/corrected"], dtype=np.float32)
        max_img = np.array(f["maxImg"], dtype=np.float32)
        mean_img = np.array(f["meanImg"], dtype=np.float32)
        roi_coords = np.array(f["rois/coords"])       # int16 (3, n_px)
        shape_arr = np.array(f["rois/shape"])          # [N, H, W]

    n_rois = int(shape_arr[0])
    img_h, img_w = int(shape_arr[1]), int(shape_arr[2])

    with h5py.File(plane_dir / "dff" / f"{plane_id}_dff.h5", "r") as f:
        dff = np.array(f["data"], dtype=np.float32)
        baseline = np.array(f["baseline"], dtype=np.float32)

    with h5py.File(
        plane_dir / "events" / f"{plane_id}_events_oasis.h5", "r"
    ) as f:
        events = np.array(f["events"], dtype=np.float32)

    with h5py.File(
        plane_dir / "classification" / f"{plane_id}_classification.h5", "r"
    ) as f:
        soma_pred = np.array(f["soma/predictions"])
        dendrite_pred = np.array(f["dendrites/predictions"])
        border = np.array(f["border/labels"])

    fs = _get_fs(session_dir, plane_id)

    return PlaneData(
        session_dir=session_dir,
        plane_id=plane_id,
        fs=fs,
        n_frames=corrected_F.shape[1],
        n_rois=n_rois,
        corrected_F=corrected_F,
        baseline=baseline,
        dff=dff,
        events=events,
        soma_pred=soma_pred,
        dendrite_pred=dendrite_pred,
        border=border,
        max_img=max_img,
        mean_img=mean_img,
        roi_coords=roi_coords,
        img_shape=(img_h, img_w),
    )
