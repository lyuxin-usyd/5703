from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    channels: int
    source_status: str
    notes: str


class RepresentationAdapter(Protocol):
    spec: AdapterSpec

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        """Return a representation tensor with shape (C, H, W)."""
        ...
