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
class GetAdapter:
    """First-pass NumPy adapter for GET event tokens.

    Upstream GET converts each event window into patch-wise token histograms with
    separate time groups, polarities, and weighted statistics. For the unified
    MVSEC benchmark we keep that grouping logic, but unwrap the patch-local
    tokens back into a dense `(C, H, W)` feature map so the result can plug into
    the shared flow head.
    """

    spec: AdapterSpec
    group_num: int = 12
    patch_size: int = 4

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        time_div = max(self.group_num // 2, 1)
        patch_h = int(self.patch_size)
        patch_w = int(self.patch_size)
        channels = time_div * 2 * 2
        rep = np.zeros((channels, height, width), dtype=np.float32)
        if events.size == 0:
            return rep

        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = _normalize_timestamps(events[:, 2])
        p = (events[:, 3] > 0).astype(np.int64)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
        t = t[valid]
        p = p[valid]
        if x.size == 0:
            return rep

        patch_x = x // patch_w
        patch_y = y // patch_h
        local_x = x % patch_w
        local_y = y % patch_h
        time_bin = np.clip((t * time_div).astype(np.int64), 0, time_div - 1)

        # Two statistics from upstream GET tokenization:
        #   stat=0 -> event count histogram
        #   stat=1 -> normalized timestamp-weighted histogram
        count_feat = np.zeros((time_div, 2, height, width), dtype=np.float32)
        time_feat = np.zeros((time_div, 2, height, width), dtype=np.float32)

        # Reconstruct patch-local token histograms at their dense pixel
        # locations. This keeps GET's patch grouping while matching the unified
        # benchmark interface expected by the downstream flow head.
        for ti, pi, px, py, lx, ly, tt in zip(time_bin, p, patch_x, patch_y, local_x, local_y, t):
            dense_x = int(px * patch_w + lx)
            dense_y = int(py * patch_h + ly)
            if dense_x >= width or dense_y >= height:
                continue
            count_feat[ti, pi, dense_y, dense_x] += 1.0
            time_feat[ti, pi, dense_y, dense_x] += float(tt)

        rep = np.concatenate(
            [
                count_feat.reshape(time_div * 2, height, width),
                time_feat.reshape(time_div * 2, height, width),
            ],
            axis=0,
        )
        return rep.astype(np.float32, copy=False)
