from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AdapterSpec


def _normalize_timestamps(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    if t.size == 0:
        return t
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max <= t_min:
        return np.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


@dataclass
class EstAdapter:
    """First-pass NumPy EST adapter.

    This is not the upstream learned MLP kernel. It follows the same shape and
    indexing logic as the EST quantization layer, but uses a trilinear kernel so
    we can integrate EST into the benchmark before torch/GPU environments are
    ready.
    """

    spec: AdapterSpec
    time_bins: int = 9

    def _trilinear_value(self, normalized_t: np.ndarray, center: float) -> np.ndarray:
        if self.time_bins <= 1:
            return np.ones_like(normalized_t, dtype=np.float32)
        delta = np.abs(normalized_t - center)
        support = 1.0 / (self.time_bins - 1)
        values = 1.0 - delta / support
        values[delta > support] = 0.0
        return values.astype(np.float32, copy=False)

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        channels = self.time_bins * 2
        rep = np.zeros((channels, height, width), dtype=np.float32)
        if events.size == 0:
            return rep

        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = _normalize_timestamps(events[:, 2])
        p = events[:, 3]

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
        t = t[valid]
        p = p[valid]

        pol = (p > 0).astype(np.int64)
        for bin_idx in range(self.time_bins):
            center = 0.0 if self.time_bins == 1 else bin_idx / (self.time_bins - 1)
            values = t * self._trilinear_value(t, center)
            for xi, yi, pi, vi in zip(x, y, pol, values):
                channel = bin_idx + pi * self.time_bins
                rep[channel, yi, xi] += float(vi)

        return rep
