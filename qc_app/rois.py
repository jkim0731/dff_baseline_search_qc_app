"""FOV image and ROI mask loading from per-session per-plane npy/pkl files."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class PlaneAssets:
    plane_id: str
    max_img:   np.ndarray    # (H, W) float32
    mean_img:  np.ndarray    # (H, W) float32
    roi_table: pd.DataFrame  # columns: cell_roi_id, mask_matrix, ...


@lru_cache(maxsize=64)
def load_plane_assets(session_path_str: str, plane_id: str) -> PlaneAssets:
    p = Path(session_path_str)
    max_img  = np.load(p / f"{plane_id}_max_img.npy").astype(np.float32)
    mean_img = np.load(p / f"{plane_id}_mean_img.npy").astype(np.float32)
    with open(p / f"{plane_id}_roi_table.pkl", "rb") as fh:
        roi_table = pickle.load(fh)
    return PlaneAssets(plane_id=plane_id, max_img=max_img,
                       mean_img=mean_img, roi_table=roi_table)


def get_roi_mask(plane: PlaneAssets, cell_roi_id: int) -> np.ndarray:
    rows = plane.roi_table[plane.roi_table["cell_roi_id"] == cell_roi_id]
    if rows.empty:
        raise IndexError(f"cell_roi_id {cell_roi_id} not in plane {plane.plane_id}")
    return rows.iloc[0]["mask_matrix"].astype(bool)


def crop_around_mask(fov: np.ndarray, mask: np.ndarray,
                     pad: int = 25) -> tuple[np.ndarray, np.ndarray, tuple]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return fov, mask, (0, 0)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, fov.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, fov.shape[1])
    return fov[y0:y1, x0:x1], mask[y0:y1, x0:x1], (y0, x0)


def normalize_for_display(img: np.ndarray,
                           lo_pct: float = 1.0,
                           hi_pct: float = 99.5) -> np.ndarray:
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
