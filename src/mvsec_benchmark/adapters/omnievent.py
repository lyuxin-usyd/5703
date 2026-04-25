from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AdapterSpec


def _normalize_timestamps(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    if t.size == 0:
        return t
    t_min, t_max = float(t.min()), float(t.max())
    if t_max <= t_min:
        return np.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


def omnievent_numpy_repr(
    events: np.ndarray,
    sensor_size: tuple[int, int],
    n_time_bins: int = 4,
) -> np.ndarray:
    """NumPy approximation of OmniEvent's decouple-enhance-fuse paradigm.

    OmniEvent separates spatial and temporal processing via PointTransformerV3
    then fuses with attention. This approximation captures the same structural
    intent using numpy:
      - Spatial channels (2): pos/neg event count maps
      - Temporal channels (n_time_bins * 2): trilinear voxel per polarity
      - Time surface channels (2): most recent timestamp per polarity (decoupled temporal)
    Total channels = 2 + n_time_bins*2 + 2 = n_time_bins*2 + 4
    """
    height, width = sensor_size
    n_ch = n_time_bins * 2 + 4
    rep = np.zeros((n_ch, height, width), dtype=np.float32)
    if events.size == 0:
        return rep

    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    t = _normalize_timestamps(events[:, 2])
    pol = (events[:, 3] > 0).astype(np.int64)  # 0=neg, 1=pos

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, pol = x[valid], y[valid], t[valid], pol[valid]
    if x.size == 0:
        return rep

    # --- Spatial channels (0,1): pos count, neg count ---
    np.add.at(rep[0], (y[pol == 1], x[pol == 1]), 1.0)
    np.add.at(rep[1], (y[pol == 0], x[pol == 0]), 1.0)

    # --- Temporal channels (2 .. 2+n_time_bins*2-1): trilinear voxel grid per polarity ---
    for b in range(n_time_bins):
        center = b / max(n_time_bins - 1, 1)
        support = 1.0 / max(n_time_bins - 1, 1)
        weights = np.maximum(0.0, 1.0 - np.abs(t - center) / support)
        ch_pos = 2 + b
        ch_neg = 2 + n_time_bins + b
        np.add.at(rep[ch_pos], (y[pol == 1], x[pol == 1]), weights[pol == 1])
        np.add.at(rep[ch_neg], (y[pol == 0], x[pol == 0]), weights[pol == 0])

    # --- Time surface channels (last 2): most recent t per polarity ---
    # Approximates OmniEvent's temporal decoupling with a surface of active events
    ts_pos = rep[-2]
    ts_neg = rep[-1]
    order = np.argsort(t)
    x_s, y_s, t_s, pol_s = x[order], y[order], t[order], pol[order]
    pos_mask = pol_s == 1
    neg_mask = pol_s == 0
    ts_pos[y_s[pos_mask], x_s[pos_mask]] = t_s[pos_mask]
    ts_neg[y_s[neg_mask], x_s[neg_mask]] = t_s[neg_mask]

    return rep


@dataclass
class OmniEventAdapter:
    """OmniEvent numpy approximation adapter.

    Captures OmniEvent's decouple-enhance-fuse idea without PointTransformerV3.
    Spatial channels (event counts) + temporal channels (voxel grid) +
    time surface channels (most recent timestamp per polarity).

    To upgrade to the real EP2T network, set use_neural=True and provide
    the cloned OmniEvent repo path after PointTransformerV3 dependencies
    are set up on AutoDL.
    """

    spec: AdapterSpec
    n_time_bins: int = 4

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        return omnievent_numpy_repr(events, sensor_size, n_time_bins=self.n_time_bins)
