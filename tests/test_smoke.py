import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.adapters import build_adapters
from mvsec_benchmark.data.synthetic import make_translation_sample
from mvsec_benchmark.models.dummy_flow import DummyFlowHead
from mvsec_benchmark.smoke import run_smoke
from mvsec_benchmark.utils.flow_metrics import compute_flow_metrics, event_valid_mask


class SmokeTest(unittest.TestCase):
    def test_all_adapters_produce_finite_outputs(self):
        sample = make_translation_sample()
        flow_head = DummyFlowHead()
        for name, adapter in build_adapters().items():
            rep = adapter.build(sample.events, sample.sensor_size)
            self.assertEqual(rep.ndim, 3, name)
            self.assertEqual(rep.shape[1:], sample.sensor_size, name)
            self.assertTrue((rep == rep).all(), name)  # no NaN
            pred = flow_head(rep)
            metrics = compute_flow_metrics(
                pred,
                sample.gt_flow,
                valid_mask=event_valid_mask(sample.events, sample.sensor_size),
            )
            self.assertTrue(math.isfinite(metrics.aee), name)
            self.assertTrue(math.isfinite(metrics.outlier_percent), name)
            self.assertGreater(metrics.valid_count, 0, name)

    def test_event_valid_mask_counts_event_pixels_only(self):
        sample = make_translation_sample(height=4, width=5, num_events=0)
        import numpy as np

        events = np.asarray(
            [
                [1, 2, 0.0, 1],
                [1, 2, 0.1, -1],
                [4, 3, 0.2, 1],
                [5, 0, 0.3, 1],
            ],
            dtype=np.float64,
        )
        mask = event_valid_mask(events, sample.sensor_size)
        self.assertEqual(int(mask.sum()), 2)
        self.assertTrue(bool(mask[2, 1]))
        self.assertTrue(bool(mask[3, 4]))

    def test_run_smoke_returns_six_methods(self):
        results = run_smoke()
        self.assertEqual(set(results.keys()), {
            "est",
            "ergo",
            "event_pretraining",
            "get",
            "matrixlstm",
            "evrepsl",
        })


if __name__ == "__main__":
    unittest.main()
