from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AdapterSpec


@dataclass
class EventPretrainingAdapter:
    """Event-frame adapter for the Event-Camera-Data-Pre-training repo.

    The upstream dataset loader uses a simple two-channel event frame:
    positive-count image and negative-count image. The pretraining paper then
    learns stronger features in the backbone rather than in a handcrafted event
    representation. For the unified MVSEC benchmark, this adapter reproduces
    that frontend exactly enough to serve as a faithful first-stage input.
    """

    spec: AdapterSpec

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        rep = np.zeros((2, height, width), dtype=np.float32)
        if events.size == 0:
            return rep

        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        p = events[:, 3].astype(np.float32)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
        p = p[valid]
        if x.size == 0:
            return rep

        pos_mask = p > 0
        neg_mask = ~pos_mask
        np.add.at(rep[0], (y[pos_mask], x[pos_mask]), 1.0)
        np.add.at(rep[1], (y[neg_mask], x[neg_mask]), 1.0)
        return rep
