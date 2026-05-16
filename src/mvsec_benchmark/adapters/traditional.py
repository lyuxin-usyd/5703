from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import AdapterSpec


def _prepare_events(events: np.ndarray, sensor_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = sensor_size
    if events.size == 0:
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_f = np.zeros((0,), dtype=np.float64)
        return empty_i, empty_i, empty_f, empty_i

    arr = np.asarray(events)
    x = arr[:, 0].astype(np.int64)
    y = arr[:, 1].astype(np.int64)
    t = arr[:, 2].astype(np.float64)
    p = arr[:, 3].astype(np.int64)

    valid = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    x, y, t, p = x[valid], y[valid], t[valid], p[valid]
    if len(x) == 0:
        return x, y, t, p

    order = np.argsort(t, kind="stable")
    x, y, t, p = x[order], y[order], t[order], p[order]
    p = np.where(p > 0, 1, 0).astype(np.int64)
    return x, y, t, p


def _polarity_channels(p: np.ndarray) -> np.ndarray:
    return np.where(p > 0, 0, 1).astype(np.int64)


def _normalize_channels(rep: np.ndarray) -> np.ndarray:
    out = rep.astype(np.float32, copy=True)
    for channel in range(out.shape[0]):
        scale = float(np.max(np.abs(out[channel])))
        if scale > 0:
            out[channel] /= scale
    return out


@dataclass
class EventFrameAdapter:
    spec: AdapterSpec
    normalize: bool = True

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        x, y, _, p = _prepare_events(events, sensor_size)
        rep = np.zeros((2, height, width), dtype=np.float32)
        if len(x):
            np.add.at(rep, (_polarity_channels(p), y, x), 1.0)
        return _normalize_channels(rep) if self.normalize else rep


@dataclass
class BinaryEventImageAdapter:
    spec: AdapterSpec

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        x, y, _, p = _prepare_events(events, sensor_size)
        rep = np.zeros((2, height, width), dtype=np.float32)
        if len(x):
            rep[_polarity_channels(p), y, x] = 1.0
        return rep


@dataclass
class TimestampImageAdapter:
    spec: AdapterSpec

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        x, y, t, p = _prepare_events(events, sensor_size)
        rep = np.zeros((2, height, width), dtype=np.float32)
        if len(x) == 0:
            return rep
        span = max(float(t[-1] - t[0]), 1.0)
        t_norm = ((t - t[0]) / span).astype(np.float32)
        np.maximum.at(rep, (_polarity_channels(p), y, x), t_norm)
        return rep


@dataclass
class TimeSurfaceAdapter:
    spec: AdapterSpec
    tau_ratio: float = 0.3

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        x, y, t, p = _prepare_events(events, sensor_size)
        latest = np.full((2, height, width), -np.inf, dtype=np.float32)
        if len(x) == 0:
            return np.zeros_like(latest, dtype=np.float32)

        t_rel = (t - float(t[0])).astype(np.float32)
        np.maximum.at(latest, (_polarity_channels(p), y, x), t_rel)
        span = max(float(t_rel[-1]), 1.0)
        tau = max(self.tau_ratio * span, 1.0)
        rep = np.zeros_like(latest, dtype=np.float32)
        active = np.isfinite(latest)
        rep[active] = np.exp(-(float(t_rel[-1]) - latest[active]) / tau).astype(np.float32)
        return rep


@dataclass
class VoxelGridAdapter:
    spec: AdapterSpec
    bins: int = 5
    normalize: bool = True

    def build(self, events: np.ndarray, sensor_size: tuple[int, int]) -> np.ndarray:
        height, width = sensor_size
        x, y, t, p = _prepare_events(events, sensor_size)
        rep = np.zeros((2 * self.bins, height, width), dtype=np.float32)
        if len(x) == 0:
            return rep

        span = max(float(t[-1] - t[0]), 1.0)
        tbin = (t - t[0]) / span * max(self.bins - 1, 1)
        lo = np.clip(np.floor(tbin).astype(np.int64), 0, self.bins - 1)
        hi = np.clip(lo + 1, 0, self.bins - 1)
        w_hi = (tbin - lo).astype(np.float32)
        w_lo = 1.0 - w_hi
        offset = _polarity_channels(p) * self.bins

        np.add.at(rep, (offset + lo, y, x), w_lo)
        np.add.at(rep, (offset + hi, y, x), w_hi)
        return _normalize_channels(rep) if self.normalize else rep
