import tempfile
import unittest
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.data import load_mvsec_windows, write_mock_mvsec_pair
from mvsec_benchmark.pipeline import run_linear_benchmark, run_torch_benchmark, run_torch_train_eval_benchmark


class PipelineTest(unittest.TestCase):
    def test_loads_official_gt_flow_dist_npz_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            flow_path = Path(tmpdir) / "indoor_flying1_gt_flow_dist.npz"
            x_flow = [[1.0, 2.0], [3.0, 4.0]]
            y_flow = [[-1.0, -2.0], [-3.0, -4.0]]
            import numpy as np

            np.savez_compressed(
                flow_path,
                timestamps=np.asarray([0.0]),
                x_flow_dist=np.asarray([x_flow], dtype=np.float32),
                y_flow_dist=np.asarray([y_flow], dtype=np.float32),
            )
            from mvsec_benchmark.data.mvsec import load_mvsec_flow

            flow = load_mvsec_flow(flow_path)
            self.assertEqual(flow.shape, (1, 2, 2, 2))
            self.assertAlmostEqual(float(flow[0, 0, 1, 0]), 2.0)
            self.assertAlmostEqual(float(flow[0, 1, 0, 1]), -3.0)

    def test_linear_benchmark_runs_on_mock_mvsec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path, flow_path = write_mock_mvsec_pair(Path(tmpdir), num_events=1000)
            windows = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
                max_windows=5,
            )
            result = run_linear_benchmark(windows, adapter_name="est", train_windows=3)
            self.assertEqual(result.adapter_name, "est")
            self.assertEqual(result.train_windows, 3)
            self.assertEqual(result.eval_windows, 2)
            self.assertGreater(result.valid_count, 0)
            self.assertTrue(result.aee == result.aee)

    def test_windows_do_not_repeat_last_flow_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path, flow_path = write_mock_mvsec_pair(Path(tmpdir), num_events=2000)
            import numpy as np

            flow = np.zeros((3, 32, 48, 2), dtype=np.float32)
            np.savez_compressed(flow_path, flow=flow)
            windows = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
            )
            self.assertEqual(len(windows), 3)

    def test_six_runnable_methods_complete_mock_suite(self):
        methods = ["est", "ergo", "event_pretraining", "get", "matrixlstm", "evrepsl"]
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path, flow_path = write_mock_mvsec_pair(Path(tmpdir), num_events=1200)
            windows = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
                max_windows=6,
            )
            for method in methods:
                result = run_linear_benchmark(windows, adapter_name=method, train_windows=4)
                self.assertEqual(result.adapter_name, method)
                self.assertGreater(result.valid_count, 0)
                self.assertTrue(result.aee == result.aee)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_benchmark_runs_on_mock_mvsec(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path, flow_path = write_mock_mvsec_pair(Path(tmpdir), num_events=1000)
            windows = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
                max_windows=5,
            )
            result = run_torch_benchmark(
                windows,
                adapter_name="est",
                train_windows=3,
                epochs=2,
                base_channels=8,
                batch_size=2,
                device="cpu",
            )
            self.assertEqual(result.adapter_name, "est")
            self.assertEqual(result.eval_windows, 2)
            self.assertGreater(result.valid_count, 0)
            self.assertTrue(result.aee == result.aee)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_benchmark_can_return_window_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path, flow_path = write_mock_mvsec_pair(Path(tmpdir), num_events=1200)
            windows = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
                max_windows=6,
            )
            result = run_torch_benchmark(
                windows,
                adapter_name="est",
                train_windows=4,
                epochs=1,
                base_channels=8,
                batch_size=2,
                eval_batch_size=1,
                device="cpu",
                return_window_metrics=True,
            )
            self.assertIsNotNone(result.window_metrics)
            self.assertEqual(len(result.window_metrics or []), result.eval_windows)
            self.assertEqual((result.window_metrics or [])[0]["sample_index"], result.train_windows)
            self.assertGreater((result.window_metrics or [])[0]["valid_count"], 0)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_train_eval_benchmark_uses_separate_sets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_h5, train_flow = write_mock_mvsec_pair(Path(tmpdir) / "train", num_events=1000)
            eval_h5, eval_flow = write_mock_mvsec_pair(Path(tmpdir) / "eval", num_events=800)
            train_samples = load_mvsec_windows(
                h5_path=train_h5,
                flow_path=train_flow,
                window_size=200,
                stride=200,
                max_windows=4,
            )
            eval_samples = load_mvsec_windows(
                h5_path=eval_h5,
                flow_path=eval_flow,
                window_size=200,
                stride=200,
                max_windows=3,
            )
            result = run_torch_train_eval_benchmark(
                train_samples,
                eval_samples,
                adapter_name="est",
                epochs=1,
                base_channels=8,
                batch_size=2,
                eval_batch_size=1,
                device="cpu",
                return_window_metrics=True,
                progress_every=0,
            )
            self.assertEqual(result.adapter_name, "est")
            self.assertEqual(result.train_windows, 4)
            self.assertEqual(result.eval_windows, 3)
            self.assertEqual(len(result.window_metrics or []), 3)
            self.assertGreater(result.valid_count, 0)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_train_eval_benchmark_supports_early_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_h5, train_flow = write_mock_mvsec_pair(Path(tmpdir) / "train", num_events=1800)
            eval_h5, eval_flow = write_mock_mvsec_pair(Path(tmpdir) / "eval", num_events=1000)
            train_samples = load_mvsec_windows(
                h5_path=train_h5,
                flow_path=train_flow,
                window_size=200,
                stride=200,
                max_windows=8,
            )
            eval_samples = load_mvsec_windows(
                h5_path=eval_h5,
                flow_path=eval_flow,
                window_size=200,
                stride=200,
                max_windows=3,
            )
            result = run_torch_train_eval_benchmark(
                train_samples,
                eval_samples,
                adapter_name="est",
                epochs=3,
                base_channels=8,
                batch_size=2,
                eval_batch_size=1,
                device="cpu",
                progress_every=0,
                early_stop_patience=2,
                early_stop_val_windows=2,
            )
            self.assertEqual(result.train_windows, 6)
            self.assertEqual(result.early_stop_val_windows, 2)
            self.assertIsNotNone(result.epochs_completed)
            self.assertIsNotNone(result.best_epoch)
            self.assertIsNotNone(result.best_val_aee)


if __name__ == "__main__":
    unittest.main()
