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

    def test_timestamp_alignment_uses_flow_time_intervals(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import h5py
            import numpy as np

            root = Path(tmpdir)
            h5_path = root / "events.h5"
            flow_path = root / "flow.npz"
            events = np.asarray(
                [
                    [1, 1, 0.0, 1],
                    [2, 2, 0.4, 1],
                    [3, 3, 0.8, 1],
                    [4, 4, 1.2, 1],
                    [5, 5, 1.6, 1],
                    [6, 6, 2.0, 1],
                ],
                dtype=np.float32,
            )
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("events", data=events)
            flow = np.zeros((3, 8, 8, 2), dtype=np.float32)
            np.savez_compressed(flow_path, flow=flow, timestamps=np.asarray([0.8, 1.6, 2.0], dtype=np.float64))

            windows = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                alignment="timestamp",
            )
            self.assertEqual(len(windows), 3)
            self.assertEqual([len(w.events) for w in windows], [2, 2, 1])
            self.assertEqual(windows[0].meta["alignment"], "timestamp")
            self.assertEqual(windows[1].meta["flow_index"], 1)
            self.assertAlmostEqual(float(windows[2].meta["flow_timestamp"]), 2.0)

    def test_timestamp_alignment_preserves_unix_second_precision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            import h5py
            import numpy as np

            root = Path(tmpdir)
            h5_path = root / "events.h5"
            flow_path = root / "flow.npz"
            base = 1_504_645_177.7612512
            events = np.asarray(
                [
                    [1, 1, base + 0.0000, 1],
                    [2, 2, base + 0.0010, 1],
                    [3, 3, base + 0.0020, 1],
                    [4, 4, base + 0.0030, 1],
                    [5, 5, base + 0.0040, 1],
                    [6, 6, base + 0.0050, 1],
                ],
                dtype=np.float64,
            )
            with h5py.File(h5_path, "w") as h5:
                h5.create_dataset("events", data=events)
            flow = np.zeros((3, 8, 8, 2), dtype=np.float32)
            timestamps = np.asarray([base + 0.0020, base + 0.0040, base + 0.0050], dtype=np.float64)
            np.savez_compressed(flow_path, flow=flow, timestamps=timestamps)

            windows = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                alignment="timestamp",
            )
            self.assertEqual([len(w.events) for w in windows], [2, 2, 1])
            self.assertEqual(windows[0].events.dtype, np.float64)
            self.assertGreater(float(windows[0].events[-1, 2] - windows[0].events[0, 2]), 0.0)

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

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_train_eval_benchmark_block_random_validation_writes_curve(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train_a_h5, train_a_flow = write_mock_mvsec_pair(root / "train_a", num_events=1400)
            train_b_h5, train_b_flow = write_mock_mvsec_pair(root / "train_b", num_events=1400)
            eval_h5, eval_flow = write_mock_mvsec_pair(root / "eval", num_events=1000)
            train_samples = []
            for h5_path, flow_path in [(train_a_h5, train_a_flow), (train_b_h5, train_b_flow)]:
                loaded = load_mvsec_windows(
                    h5_path=h5_path,
                    flow_path=flow_path,
                    window_size=200,
                    stride=200,
                    max_windows=5,
                )
                for sample in loaded:
                    sample.meta["source_h5"] = str(h5_path)
                    sample.meta["source_flow"] = str(flow_path)
                train_samples.extend(loaded)
            eval_samples = load_mvsec_windows(
                h5_path=eval_h5,
                flow_path=eval_flow,
                window_size=200,
                stride=200,
                max_windows=3,
            )
            curve_path = root / "curves" / "run.csv"
            result = run_torch_train_eval_benchmark(
                train_samples,
                eval_samples,
                adapter_name="est",
                epochs=2,
                base_channels=8,
                batch_size=2,
                eval_batch_size=1,
                device="cpu",
                progress_every=0,
                early_stop_patience=2,
                early_stop_val_windows=4,
                early_stop_val_strategy="block-random",
                curve_log_path=curve_path,
            )
            self.assertEqual(result.early_stop_val_strategy, "block-random")
            self.assertEqual(result.early_stop_val_windows, 4)
            self.assertIsNotNone(result.early_stop_val_source_counts)
            self.assertEqual(sum((result.early_stop_val_source_counts or {}).values()), 4)
            self.assertTrue(curve_path.exists())
            lines = curve_path.read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual(lines[0], "epoch,train_loss,val_aee,best_val_aee,is_best,stale_epochs,early_stopped")

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_train_eval_benchmark_accepts_image_pair_regularizers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path, flow_path = write_mock_mvsec_pair(Path(tmpdir), num_events=1000)
            samples = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
                max_windows=4,
            )
            import numpy as np

            for sample in samples:
                height, width = sample.sensor_size
                sample.meta["sequence"] = "indoor_flying1"
                object.__setattr__(sample, "prev_image", np.zeros((height, width), dtype=np.float32))
                object.__setattr__(sample, "next_image", np.zeros((height, width), dtype=np.float32))

            result = run_torch_train_eval_benchmark(
                samples[:3],
                samples[3:],
                adapter_name="est",
                epochs=1,
                base_channels=4,
                batch_size=2,
                eval_batch_size=1,
                device="cpu",
                progress_every=0,
                learning_rate=3e-4,
                photometric_weight=1.0,
                smoothness_weight=0.5,
                photometric_loss="evflownet",
                photometric_charbonnier_alpha=0.45,
                smoothness_mode="evflownet_8conn",
                lr_schedule="evflownet",
                lr_decay=0.9,
                weight_decay=1e-4,
            )
            self.assertEqual(result.train_windows, 3)
            self.assertEqual(result.eval_windows, 1)
            self.assertGreater(result.valid_count, 0)
            self.assertIsNotNone(result.per_sequence_metrics)
            self.assertIn("indoor_flying1", result.per_sequence_metrics or {})

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_train_eval_benchmark_accepts_multiscale_self_supervised(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path, flow_path = write_mock_mvsec_pair(Path(tmpdir), num_events=1000)
            samples = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
                max_windows=4,
            )
            import numpy as np

            for sample in samples:
                height, width = sample.sensor_size
                sample.meta["sequence"] = "indoor_flying1"
                object.__setattr__(sample, "prev_image", np.zeros((height, width), dtype=np.float32))
                object.__setattr__(sample, "next_image", np.zeros((height, width), dtype=np.float32))

            result = run_torch_train_eval_benchmark(
                samples[:3],
                samples[3:],
                adapter_name="est",
                epochs=1,
                base_channels=8,
                model_variant="evflownet_multiscale",
                training_objective="self_supervised",
                supervised_weight=0.0,
                batch_size=2,
                eval_batch_size=1,
                device="cpu",
                progress_every=0,
                learning_rate=3e-4,
                photometric_weight=1.0,
                smoothness_weight=0.5,
                photometric_loss="evflownet",
                photometric_use_valid_mask=True,
                smoothness_mode="evflownet_8conn",
                lr_schedule="evflownet",
            )
            self.assertEqual(result.train_windows, 3)
            self.assertEqual(result.eval_windows, 1)
            self.assertGreater(result.valid_count, 0)
            self.assertIn("indoor_flying1", result.per_sequence_metrics or {})

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_train_eval_benchmark_accepts_batchnorm_crop_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path, flow_path = write_mock_mvsec_pair(Path(tmpdir), num_events=1200)
            samples = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
                max_windows=5,
            )
            import numpy as np

            for sample in samples:
                height, width = sample.sensor_size
                sample.meta["sequence"] = "indoor_flying1"
                object.__setattr__(sample, "prev_image", np.zeros((height, width), dtype=np.float32))
                object.__setattr__(sample, "next_image", np.zeros((height, width), dtype=np.float32))

            result = run_torch_train_eval_benchmark(
                samples[:4],
                samples[4:],
                adapter_name="est",
                epochs=1,
                base_channels=8,
                model_variant="evflownet_multiscale",
                training_objective="self_supervised",
                supervised_weight=0.0,
                batch_size=2,
                eval_batch_size=1,
                device="cpu",
                progress_every=0,
                learning_rate=3e-4,
                photometric_weight=1.0,
                smoothness_weight=0.5,
                photometric_loss="evflownet",
                smoothness_mode="evflownet_8conn",
                lr_schedule="evflownet",
                model_batch_norm=True,
                paper_crop_size=32,
                paper_train_random_crop=True,
                paper_random_flip=True,
                paper_random_rotation_degrees=5.0,
                metric_scope="evflownet_official",
            )
            self.assertEqual(result.train_windows, 4)
            self.assertEqual(result.eval_windows, 1)
            self.assertGreater(result.valid_count, 0)
            self.assertIn("indoor_flying1", result.per_sequence_metrics or {})

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed in this interpreter")
    def test_torch_train_eval_benchmark_exports_checkpoint_and_inference_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            h5_path, flow_path = write_mock_mvsec_pair(root, num_events=1200)
            samples = load_mvsec_windows(
                h5_path=h5_path,
                flow_path=flow_path,
                window_size=200,
                stride=200,
                max_windows=5,
            )
            import numpy as np

            for sample in samples:
                height, width = sample.sensor_size
                sample.meta["sequence"] = "indoor_flying1"
                sample.meta["source_h5"] = str(h5_path)
                sample.meta["source_flow"] = str(flow_path)
                object.__setattr__(sample, "prev_image", np.zeros((height, width), dtype=np.float32))
                object.__setattr__(sample, "next_image", np.zeros((height, width), dtype=np.float32))

            artifact_root = root / "artifacts" / "est"
            result = run_torch_train_eval_benchmark(
                samples[:4],
                samples[4:],
                adapter_name="est",
                epochs=1,
                base_channels=8,
                model_variant="evflownet_multiscale",
                training_objective="self_supervised",
                supervised_weight=0.0,
                batch_size=2,
                eval_batch_size=1,
                device="cpu",
                progress_every=0,
                learning_rate=3e-4,
                photometric_weight=1.0,
                smoothness_weight=0.5,
                photometric_loss="evflownet",
                smoothness_mode="evflownet_8conn",
                lr_schedule="evflownet",
                checkpoint_path=artifact_root / "best_checkpoint.pt",
                inference_dir=artifact_root / "inference",
                inference_sample_indices=[0],
                inference_manifest_path=artifact_root / "sample_manifest.csv",
            )
            self.assertEqual(result.checkpoint_path, str(artifact_root / "best_checkpoint.pt"))
            self.assertTrue((artifact_root / "best_checkpoint.pt").exists())
            self.assertTrue((artifact_root / "sample_manifest.csv").exists())
            exported_samples = list((artifact_root / "inference").glob("sample_*"))
            self.assertEqual(len(exported_samples), 1)
            self.assertTrue((exported_samples[0] / "pred_flow.npz").exists())
            self.assertTrue((exported_samples[0] / "pred_flow.png").exists())
            self.assertTrue((exported_samples[0] / "sample_meta.json").exists())


if __name__ == "__main__":
    unittest.main()
