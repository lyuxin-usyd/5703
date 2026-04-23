from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AdapterSpec


def _aggregate_to_image(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    height: int,
    width: int,
    aggregation: str,
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.float32)
    if x.size == 0:
        return out

    index = y.astype(np.int64) * width + x.astype(np.int64)
    flat = np.zeros(height * width, dtype=np.float32)

    if aggregation == "sum":
        np.add.at(flat, index, values.astype(np.float32))
    elif aggregation == "mean":
        count = np.zeros(height * width, dtype=np.float32)
        np.add.at(flat, index, values.astype(np.float32))
        np.add.at(count, index, 1.0)
        valid = count > 0
        flat[valid] /= count[valid]
    elif aggregation == "max":
        flat[:] = -np.inf
        np.maximum.at(flat, index, values.astype(np.float32))
        flat[~np.isfinite(flat)] = 0.0
    elif aggregation == "variance":
        s1 = np.zeros(height * width, dtype=np.float32)
        s2 = np.zeros(height * width, dtype=np.float32)
        count = np.zeros(height * width, dtype=np.float32)
        np.add.at(s1, index, values.astype(np.float32))
        np.add.at(s2, index, values.astype(np.float32) ** 2)
        np.add.at(count, index, 1.0)
        valid = count > 0
        flat[valid] = np.maximum(s2[valid] / count[valid] - (s1[valid] / count[valid]) ** 2, 0.0)
    else:
        raise ValueError(f"Unsupported aggregation: {aggregation}")

    return flat.reshape(height, width)


def _surface_from_events(
    events: np.ndarray,
    height: int,
    width: int,
    func: str,
    aggregation: str,
) -> np.ndarray:
    x = events[:, 0].astype(np.int64)
    y = events[:, 1].astype(np.int64)
    t = events[:, 2].astype(np.float32)
    p = events[:, 3].astype(np.int8)

    if func == "timestamp":
        values = t
        mask = np.ones_like(t, dtype=bool)
    elif func == "polarity":
        values = p.astype(np.float32)
        mask = np.ones_like(t, dtype=bool)
    elif func == "count":
        values = np.ones_like(t, dtype=np.float32)
        mask = np.ones_like(t, dtype=bool)
    elif func == "timestamp_pos":
        values = t
        mask = p > 0
    elif func == "timestamp_neg":
        values = t
        mask = p <= 0
    elif func == "count_pos":
        values = np.ones_like(t, dtype=np.float32)
        mask = p > 0
    elif func == "count_neg":
        values = np.ones_like(t, dtype=np.float32)
        mask = p <= 0
    else:
        raise ValueError(f"Unsupported ERGO function: {func}")

    return _aggregate_to_image(x[mask], y[mask], values[mask], height, width, aggregation)


def _normalize_timestamps(t: np.ndarray) -> np.ndarray:
    if t.size == 0:
        return t.astype(np.float32)
    t = t.astype(np.float32)
    t_min = float(t.min())
    t_max = float(t.max())
    if t_max <= t_min:
        return np.zeros_like(t)
    return (t - t_min) / (t_max - t_min)


@dataclass
class ErgoAdapter:
    """NumPy ERGO-style optimized representation adapter.

    This follows the published optimized-representation recipe from the upstream
    repo, but implemented without torch_scatter so it can run locally in the
    benchmark scaffold.
    """

    spec: AdapterSpec

    window_indexes = [0, 3, 2, 6, 5, 6, 2, 5, 1, 0, 4, 1]
    functions = [
        "polarity",
        "timestamp_neg",
        "count_neg",
        "polarity",
        "count_pos",
        "count",
        "timestamp_pos",
        "count_neg",
        "timestamp_neg",
        "timestamp_pos",
        "timestamp",
        "count",
    ]
    aggregations = [
        "variance",
        "variance",
        "mean",
        "sum",
        "mean",
        "sum",
        "mean",
        "mean",
        "max",
        "max",
        "max",
        "mean",
    ]

    def _create_windows(self, events: np.ndarray) -> list[np.ndarray]:
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = _normalize_timestamps(events[:, 2])
        p = np.where(events[:, 3] > 0, 1, -1).astype(np.int8)
        base = np.stack([x, y, t, p], axis=1)

        windows: list[np.ndarray] = [base]
        n = len(base)
        third = max(n // 3, 1)

        # Equispaced event-count windows.
        for i in range(3):
            start = i * third
            end = n if i == 2 else min((i + 1) * third, n)
            windows.append(base[start:end])

        # Halving suffix windows.
        cur = base
        for _ in range(3):
            cut = max(len(cur) // 2, 1)
            cur = cur[cut:]
            windows.append(cur)

        return windows

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        rep = np.zeros((12, height, width), dtype=np.float32)
        if events.size == 0:
            return rep

        valid = (
            (events[:, 0] >= 0)
            & (events[:, 0] < width)
            & (events[:, 1] >= 0)
            & (events[:, 1] < height)
        )
        events = events[valid]
        windows = self._create_windows(events)

        for i, (widx, func, agg) in enumerate(zip(self.window_indexes, self.functions, self.aggregations)):
            window_events = windows[widx] if widx < len(windows) else np.empty((0, 4), dtype=np.float32)
            rep[i] = _surface_from_events(window_events, height, width, func, agg)

        return rep
