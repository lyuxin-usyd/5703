from __future__ import annotations

from .adapters import build_adapters
from .data.synthetic import make_translation_sample
from .models.dummy_flow import DummyFlowHead
from .utils.flow_metrics import compute_flow_metrics


def run_smoke() -> dict[str, dict[str, float]]:
    sample = make_translation_sample()
    flow_head = DummyFlowHead()
    results: dict[str, dict[str, float]] = {}
    for name, adapter in build_adapters().items():
        rep = adapter.build(sample.events, sample.sensor_size)
        pred = flow_head(rep)
        metrics = compute_flow_metrics(pred, sample.gt_flow, outlier_mode="kitti")
        results[name] = {
            "channels": float(rep.shape[0]),
            "aee": metrics.aee,
            "outlier_percent": metrics.outlier_percent,
            "valid_count": float(metrics.valid_count),
        }
    return results
