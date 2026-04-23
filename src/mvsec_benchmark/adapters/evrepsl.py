from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import AdapterSpec


def events_to_evrep(
    event_xs: np.ndarray,
    event_ys: np.ndarray,
    event_timestamps: np.ndarray,
    event_polarities: np.ndarray,
    resolution: tuple[int, int],
) -> np.ndarray:
    """Pure NumPy EvRep implementation adapted from the upstream EvRepSL repo."""
    width, height = resolution

    e_c = np.zeros((height, width), dtype=np.float32)
    e_i = np.zeros((height, width), dtype=np.float32)
    e_t_sum = np.zeros((height, width), dtype=np.float32)
    e_t_sq_sum = np.zeros((height, width), dtype=np.float32)

    polarities = np.where(event_polarities == 0, -1, event_polarities).astype(np.float32)

    np.add.at(e_c, (event_ys, event_xs), 1.0)
    np.add.at(e_i, (event_ys, event_xs), polarities)

    sort_idx = np.lexsort((event_timestamps, event_ys, event_xs))
    sorted_x = event_xs[sort_idx]
    sorted_y = event_ys[sort_idx]
    sorted_t = event_timestamps[sort_idx]

    if sorted_t.size:
        dt = np.diff(sorted_t, prepend=sorted_t[0]).astype(np.float32)
        np.add.at(e_t_sum, (sorted_y, sorted_x), dt)
        np.add.at(e_t_sq_sum, (sorted_y, sorted_x), dt ** 2)

    counts = np.clip(e_c, a_min=1.0, a_max=None)
    dt_mean = e_t_sum / counts
    e_t = np.sqrt(np.maximum((e_t_sq_sum / counts) - dt_mean**2, 0.0))
    return np.stack([e_c, e_i, e_t], axis=0).astype(np.float32, copy=False)


@dataclass
class EvRepSLAdapter:
    """EvRepSL adapter with a graceful fallback path.

    Current stage:
    - always supports EvRep generation in NumPy
    - optionally upgrades to EvRepSL if torch and RepGen weights are available
    """

    spec: AdapterSpec
    repgen_weights: str | None = None
    device: str = "cpu"

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        if events.size == 0:
            return np.zeros((3, height, width), dtype=np.float32)

        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = events[:, 2].astype(np.float32)
        p = (events[:, 3] > 0).astype(np.int64)

        valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x = x[valid]
        y = y[valid]
        t = t[valid]
        p = p[valid]

        ev_rep = events_to_evrep(x, y, t, p, (width, height))
        if not self.repgen_weights:
            return ev_rep

        weights = Path(self.repgen_weights)
        if not weights.exists():
            return ev_rep

        try:
            import torch

            # Local import from the cloned upstream repo to avoid copying their code into the
            # benchmark package at this early integration stage.
            import importlib.util

            upstream = Path(__file__).resolve().parents[3] / "refs" / "EvRepSL" / "models.py"
            spec = importlib.util.spec_from_file_location("evrepsl_upstream_models", upstream)
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            spec.loader.exec_module(module)
            model = module.EffWNet(
                n_channels=3,
                out_depth=1,
                inc_f0=1,
                bilinear=True,
                n_lyr=4,
                ch1=12,
                c_is_const=False,
                c_is_scalar=False,
                device=self.device,
            )
            state = torch.load(str(weights), map_location=self.device)
            model.load_state_dict(state)
            model.eval()
            with torch.no_grad():
                inp = torch.tensor(ev_rep[None, ...], dtype=torch.float32, device=self.device)
                out = model(inp).detach().cpu().numpy()[0]
            return out.astype(np.float32, copy=False)
        except Exception:
            return ev_rep
