from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class FlowMetrics:
    aee: float
    outlier_percent: float
    valid_count: int
    outlier_count: int


def ensure_hw2(flow: np.ndarray) -> np.ndarray:
    arr = np.asarray(flow, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D flow array, got {arr.shape}")
    if arr.shape[-1] == 2:
        return arr
    if arr.shape[0] == 2:
        return np.moveaxis(arr, 0, -1)
    raise ValueError(f"Expected flow shape (H,W,2) or (2,H,W), got {arr.shape}")


def compute_flow_metrics(
    pred_flow: np.ndarray,
    gt_flow: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    outlier_mode: str = "kitti",
) -> FlowMetrics:
    pred = ensure_hw2(pred_flow)
    gt = ensure_hw2(gt_flow)
    epe = np.linalg.norm(pred - gt, axis=-1)
    valid = np.isfinite(epe) & np.isfinite(gt[..., 0]) & np.isfinite(gt[..., 1])
    if valid_mask is not None:
        valid &= np.asarray(valid_mask).astype(bool)
    valid_count = int(valid.sum())
    if valid_count == 0:
        return FlowMetrics(float("nan"), float("nan"), 0, 0)

    gt_mag = np.linalg.norm(gt, axis=-1)
    if outlier_mode == "kitti":
        outlier = (epe > 3.0) & (epe > 0.05 * gt_mag) & valid
    elif outlier_mode == "px":
        outlier = (epe > 3.0) & valid
    else:
        raise ValueError("outlier_mode must be 'kitti' or 'px'")

    outlier_count = int(outlier.sum())
    return FlowMetrics(
        aee=float(epe[valid].mean()),
        outlier_percent=100.0 * outlier_count / valid_count,
        valid_count=valid_count,
        outlier_count=outlier_count,
    )
