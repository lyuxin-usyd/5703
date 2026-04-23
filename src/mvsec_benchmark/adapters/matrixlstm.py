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
class MatrixLSTMAdapter:
    """Per-pixel sequence adapter inspired by MatrixLSTM optical-flow defaults.

    The official optical-flow setup defaults to a 1x1 receptive field and a
    four-channel output. This adapter keeps that spirit without pulling in the
    old TensorFlow/CUDA grouping kernels: events are grouped per pixel, ordered
    in time, and summarized into four dense sequence features that mimic a
    lightweight recurrent state.
    """

    spec: AdapterSpec
    tau: float = 0.25

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        rep = np.zeros((4, height, width), dtype=np.float32)
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

        pixel_id = y * width + x
        order = np.lexsort((t, pixel_id))
        pixel_id = pixel_id[order]
        t = t[order]
        p = p[order]

        state = np.zeros(height * width, dtype=np.float32)
        last_t = np.zeros(height * width, dtype=np.float32)
        delay_sum = np.zeros(height * width, dtype=np.float32)
        count = np.zeros(height * width, dtype=np.float32)
        last_p = np.zeros(height * width, dtype=np.float32)

        for pid, ti, pi in zip(pixel_id, t, p):
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

        rep[0] = state.reshape(height, width)
        rep[1] = last_t.reshape(height, width)
        rep[2] = mean_delay.reshape(height, width)
        rep[3] = last_p.reshape(height, width)
        return rep
