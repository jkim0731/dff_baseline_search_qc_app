"""ROI mask building and FOV crop helpers for production h5 data."""

from __future__ import annotations

import numpy as np

from .data import PlaneData

from .data import PlaneData


def get_roi_mask(plane: PlaneData, roi_idx: int) -> np.ndarray:
    """Binary (H, W) mask from sparse coords stored in extraction.h5."""
    coords = plane.roi_coords
    px = np.where(coords[0] == roi_idx)[0]
    mask = np.zeros(plane.img_shape, dtype=bool)
    if len(px):
        ys = coords[1][px].astype(int)
        xs = coords[2][px].astype(int)
        mask[ys, xs] = True
    return mask


def crop_around_mask(
    fov: np.ndarray, mask: np.ndarray, pad: int = 40
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return fov, mask, (0, 0)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, fov.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, fov.shape[1])
    return fov[y0:y1, x0:x1], mask[y0:y1, x0:x1], (y0, x0)


def normalize_img(
    img: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.5
) -> np.ndarray:
    finite = img[np.isfinite(img)]
    if len(finite) == 0:
        return np.zeros_like(img, dtype=np.float32)
    lo, hi = np.percentile(finite, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((img.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)


def mask_contour(mask: np.ndarray) -> np.ndarray:
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return mask & ~interior
