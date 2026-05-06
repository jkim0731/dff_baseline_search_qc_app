"""FOV image and ROI mask loading from extraction h5 files."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import h5py
import numpy as np
import sparse


@dataclass
class PlaneAssets:
    plane_id: str
    fov: np.ndarray        # (H, W) float32
    masks: np.ndarray      # (n_rois, H, W) bool
    cell_roi_ids: np.ndarray


@lru_cache(maxsize=64)
def load_plane_assets(plane_path_str: str) -> PlaneAssets:
    plane_path = Path(plane_path_str)
    plane_id = plane_path.name
    h5_path = plane_path / "extraction" / f"{plane_id}_extraction.h5"
    with h5py.File(h5_path, "r") as h:
        fov = h["maxImg"][:]
        coords = h["rois"]["coords"][:]
        data   = h["rois"]["data"][:]
        shape  = tuple(h["rois"]["shape"][:])
    masks = sparse.COO(coords, data, shape).todense() > 0
    return PlaneAssets(plane_id=plane_id, fov=fov.astype(np.float32),
                       masks=masks, cell_roi_ids=np.arange(masks.shape[0]))


def get_roi_mask(plane: PlaneAssets, cell_roi_id: int) -> np.ndarray:
    if cell_roi_id < 0 or cell_roi_id >= len(plane.masks):
        raise IndexError(f"cell_roi_id {cell_roi_id} out of range")
    return plane.masks[cell_roi_id]


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
