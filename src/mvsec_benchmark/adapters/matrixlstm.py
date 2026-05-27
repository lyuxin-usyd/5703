from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AdapterSpec


def _normalize_timestamps(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    if t.size == 0:
        return t.astype(np.float32)
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max <= t_min:
        return np.zeros_like(t, dtype=np.float32)
    return ((t - t_min) / (t_max - t_min)).astype(np.float32)


@dataclass
class MatrixLSTMAdapter:
    """Per-pixel sequence adapter inspired by MatrixLSTM optical-flow defaults.

    The official optical-flow setup uses a 1x1 receptive field. The paper's
    strongest optical-flow ablation is the two-bin surface, so this adapter
    emits four recurrent-summary channels per temporal bin: events are grouped
    per pixel, ordered in time, and summarized into dense sequence features
    that mimic a lightweight recurrent state.
    """

    spec: AdapterSpec
    tau: float = 0.25
    time_bins: int = 2

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        bins = max(int(self.time_bins), 1)
        rep = np.zeros((4 * bins, height, width), dtype=np.float32)
        if events.size == 0:
            return rep

        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = _normalize_timestamps(events[:, 2])
        p = np.where(events[:, 3] > 0, 1.0, -1.0).astype(np.float32)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
        t = t[valid]
        p = p[valid]
        if x.size == 0:
            return rep

        if bins == 1:
            bin_ids = np.zeros_like(t, dtype=np.int64)
        else:
            bin_ids = np.minimum((t * bins).astype(np.int64), bins - 1)

        for bin_idx in range(bins):
            in_bin = bin_ids == bin_idx
            if not np.any(in_bin):
                continue

            xb = x[in_bin]
            yb = y[in_bin]
            tb = t[in_bin]
            pb = p[in_bin]
            if bins > 1:
                start = bin_idx / bins
                stop = (bin_idx + 1) / bins
                tb = ((tb - start) / max(stop - start, 1e-6)).astype(np.float32)
            pixel_id = yb * width + xb
            order = np.lexsort((tb, pixel_id))
            pixel_id = pixel_id[order]
            tb = tb[order]
            pb = pb[order]

            state = np.zeros(height * width, dtype=np.float32)
            last_t = np.zeros(height * width, dtype=np.float32)
            delay_sum = np.zeros(height * width, dtype=np.float32)
            count = np.zeros(height * width, dtype=np.float32)
            last_p = np.zeros(height * width, dtype=np.float32)

            for pid, ti, pi in zip(pixel_id, tb, pb):
                dt = float(ti - last_t[pid]) if count[pid] > 0 else 0.0
                decay = np.exp(-dt / max(self.tau, 1e-6))
                state[pid] = state[pid] * decay + pi
                last_t[pid] = float(ti)
                delay_sum[pid] += dt
                count[pid] += 1.0
                last_p[pid] = pi

            valid_pixels = count > 0
            mean_delay = np.zeros_like(delay_sum)
            mean_delay[valid_pixels] = delay_sum[valid_pixels] / count[valid_pixels]
            offset = bin_idx * 4
            rep[offset + 0] = state.reshape(height, width)
            rep[offset + 1] = last_t.reshape(height, width)
            rep[offset + 2] = mean_delay.reshape(height, width)
            rep[offset + 3] = last_p.reshape(height, width)
        return rep
