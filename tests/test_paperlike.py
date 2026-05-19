import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h5py
import numpy as np

from mvsec_benchmark.paperlike import (
    compute_matrixlstm_paperlike_metrics,
    estimate_corresponding_gt_flow,
    load_matrixlstm_paperlike_windows,
)
from mvsec_benchmark.data.mvsec import FlowWindowSample


class PaperLikeTest(unittest.TestCase):
    def test_estimate_corresponding_gt_flow_scales_short_interval(self):
        flow = np.zeros((2, 4, 5, 2), dtype=np.float32)
        flow[1, ..., 0] = 2.0
        timestamps = np.asarray([0.0, 1.0], dtype=np.float64)

        estimated = estimate_corresponding_gt_flow(flow, timestamps, 0.0, 0.5)

        self.assertEqual(estimated.shape, (4, 5, 2))
        self.assertTrue(np.allclose(estimated[..., 0], 1.0))
        self.assertTrue(np.allclose(estimated[..., 1], 0.0))

    def test_paperlike_metrics_count_event_and_nonzero_gt_pixels_only(self):
        events = np.asarray([[1, 1, 0.1, 1], [2, 1, 0.2, -1]], dtype=np.float64)
        gt = np.zeros((4, 5, 2), dtype=np.float32)
        gt[1, 1, 0] = 1.0
        gt[1, 2, 0] = 0.0
        pred = np.zeros_like(gt)
        pred[1, 1, 0] = 2.0
        pred[0, 0, 0] = 100.0
        sample = FlowWindowSample(events=events, gt_flow=gt, sensor_size=(4, 5))

        metrics = compute_matrixlstm_paperlike_metrics(pred, sample)

        self.assertEqual(metrics.valid_count, 1)
        self.assertAlmostEqual(metrics.aee, 1.0)

    def test_load_matrixlstm_paperlike_windows_propagates_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            h5_path = root / "events.h5"
            flow_path = root / "flow.npz"
            events = np.asarray(
                [
                    [1, 1, 0.1, 1],
                    [2, 2, 0.4, 1],
                    [3, 3, 0.8, -1],
                    [4, 4, 1.2, 1],
                ],
                dtype=np.float64,
            )
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("events", data=events)

            flow = np.zeros((3, 8, 8, 2), dtype=np.float32)
            flow[1, ..., 0] = 2.0
            flow[2, ..., 1] = 4.0
            np.savez_compressed(flow_path, flow=flow, timestamps=np.asarray([0.0, 1.0, 2.0]))

            windows = load_matrixlstm_paperlike_windows(h5_path, flow_path)

            self.assertEqual(len(windows), 2)
            self.assertEqual(windows[0].meta["alignment"], "matrixlstm_paperlike")
            self.assertTrue(np.allclose(windows[0].gt_flow[..., 0], 2.0))
            self.assertTrue(np.allclose(windows[1].gt_flow[..., 1], 4.0))


if __name__ == "__main__":
    unittest.main()
