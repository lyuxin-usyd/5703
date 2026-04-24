from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticFlowSample:
    events: np.ndarray
    gt_flow: np.ndarray
    sensor_size: tuple[int, int]


def make_translation_sample(
    height: int = 32,
    width: int = 48,
    num_events: int = 400,
    flow_xy: tuple[float, float] = (1.5, -0.5),
    seed: int = 42,
) -> SyntheticFlowSample:
    """Create a small synthetic event stream and constant GT flow field.

    Event layout follows a common convention: columns are [x, y, t, p].
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(0, width, size=num_events)
    y = rng.integers(0, height, size=num_events)
    t = np.sort(rng.random(size=num_events).astype(np.float32))
    p = rng.choice([-1.0, 1.0], size=num_events).astype(np.float32)
    events = np.stack([x, y, t, p], axis=1).astype(np.float32)

    gt_flow = np.zeros((height, width, 2), dtype=np.float32)
    gt_flow[..., 0] = flow_xy[0]
    gt_flow[..., 1] = flow_xy[1]
    return SyntheticFlowSample(events=events, gt_flow=gt_flow, sensor_size=(height, width))
