from __future__ import annotations

import numpy as np


def _normalize_timestamps(events: np.ndarray) -> np.ndarray:
    t = events[:, 2].astype(np.float32)
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max <= t_min:
        return np.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


def voxel_count_representation(
    events: np.ndarray,
    sensor_size: tuple[int, int],
    time_bins: int,
    split_polarity: bool = True,
) -> np.ndarray:
    """Simple event voxelization used as a placeholder adapter backend.

    This is benchmark scaffolding, not a faithful implementation of each paper.
    """
    height, width = sensor_size
    channels = time_bins * (2 if split_polarity else 1)
    rep = np.zeros((channels, height, width), dtype=np.float32)
    if events.size == 0:
        return rep

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    t = _normalize_timestamps(events)
    p = events[:, 3]

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    t = t[valid]
    p = p[valid]

    bins = np.clip((t * time_bins).astype(np.int64), 0, time_bins - 1)
    if split_polarity:
        pol = (p > 0).astype(np.int64)
        c = bins * 2 + pol
    else:
        c = bins

    for ci, yi, xi, pi in zip(c, y, x, p):
        rep[ci, yi, xi] += float(pi)
    return rep


def recurrent_surface_representation(
    events: np.ndarray,
    sensor_size: tuple[int, int],
    tau: float = 0.25,
    channels: int = 4,
) -> np.ndarray:
    """Small recurrent-surface approximation for MatrixLSTM placeholder work."""
    height, width = sensor_size
    rep = np.zeros((channels, height, width), dtype=np.float32)
    if events.size == 0:
        return rep

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    t = _normalize_timestamps(events)
    p = events[:, 3]
    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x = x[valid]
    y = y[valid]
    t = t[valid]
    p = p[valid]

    last_t = np.zeros((height, width), dtype=np.float32)
    acc = np.zeros((height, width), dtype=np.float32)
    for xi, yi, ti, pi in zip(x, y, t, p):
        dt = ti - last_t[yi, xi]
        acc[yi, xi] = acc[yi, xi] * np.exp(-dt / max(tau, 1e-6)) + pi
        last_t[yi, xi] = ti

    rep[0] = acc
    rep[1] = last_t
    rep[2] = np.maximum(acc, 0.0)
    rep[3] = np.maximum(-acc, 0.0)
    return rep
