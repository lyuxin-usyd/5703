from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from .synthetic import make_translation_sample


def write_mock_mvsec_pair(
    root: str | Path,
    name: str = "mock_mvsec",
    *,
    height: int = 32,
    width: int = 48,
    num_events: int = 1200,
    flow_xy: tuple[float, float] = (1.5, -0.5),
    seed: int = 42,
) -> tuple[Path, Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    sample = make_translation_sample(
        height=height,
        width=width,
        num_events=num_events,
        flow_xy=flow_xy,
        seed=seed,
    )

    h5_path = root / f"{name}_data.h5"
    flow_path = root / f"{name}_flow.npz"

    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("events", data=sample.events.astype(np.float32))

    np.savez_compressed(flow_path, flow=sample.gt_flow.astype(np.float32))
    return h5_path, flow_path
