import tempfile
import unittest
from pathlib import Path

from mvsec_benchmark.data import load_mvsec_windows, write_mock_mvsec_pair
from mvsec_benchmark.pipeline import run_linear_benchmark


class PipelineTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
