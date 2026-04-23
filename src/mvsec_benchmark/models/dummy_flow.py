from __future__ import annotations

import numpy as np


class DummyFlowHead:
    """Tiny deterministic flow head for local smoke tests.

    This is not a training model. It just converts a representation tensor into
    a finite flow map so the benchmark interface can be validated without torch
    or MVSEC downloads.
    """

    def __call__(self, representation: np.ndarray) -> np.ndarray:
        rep = np.asarray(representation, dtype=np.float32)
        if rep.ndim != 3:
            raise ValueError(f"Expected representation shape (C,H,W), got {rep.shape}")
        mean_map = rep.mean(axis=0)
        max_map = rep.max(axis=0)
        flow = np.stack([mean_map, max_map - mean_map], axis=-1)
        return flow.astype(np.float32, copy=False)
