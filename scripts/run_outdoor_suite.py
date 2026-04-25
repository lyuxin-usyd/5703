"""Legacy outdoor-train / indoor-eval benchmark helper.

Train on: outdoor_day1 + outdoor_day2
Eval on:  indoor_flying1, indoor_flying2, indoor_flying3 (separately)

This script is kept for reference only. The current formal reproduction path is
`scripts/run_original_protocol.py`, or `scripts/autodl_outdoor_pipeline.sh` on
AutoDL. The legacy argument shape below predates the successful 6M-event /
full-GT-flow protocol and should not be used for the report unless debugging.

Usage example:
  python scripts/run_outdoor_suite.py \
    --adapter est \
    --train-h5  data/mvsec/outdoor_day/outdoor_day1_left_events.h5 \
                data/mvsec/outdoor_day/outdoor_day2_left_events.h5 \
    --train-flow data/mvsec/outdoor_day/outdoor_day1_gt_flow.npz \
                 data/mvsec/outdoor_day/outdoor_day2_gt_flow.npz \
    --eval-h5   data/mvsec/indoor_flying/indoor_flying1_left_events.h5 \
                data/mvsec/indoor_flying/indoor_flying2_left_events.h5 \
                data/mvsec/indoor_flying/indoor_flying3_left_events.h5 \
    --eval-flow data/mvsec/indoor_flying/indoor_flying1_gt_flow.npz \
                data/mvsec/indoor_flying/indoor_flying2_gt_flow.npz \
                data/mvsec/indoor_flying/indoor_flying3_gt_flow.npz \
    --epochs 50 --device cuda --disable-cudnn \
    --output results/outdoor_suite_est.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mvsec_benchmark.data import load_mvsec_windows
from mvsec_benchmark.adapters import build_adapters
from mvsec_benchmark.utils.flow_metrics import compute_flow_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Outdoor-train / indoor-eval MVSEC benchmark.")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--train-h5", nargs="+", type=Path, required=True)
    parser.add_argument("--train-flow", nargs="+", type=Path, required=True)
    parser.add_argument("--eval-h5", nargs="+", type=Path, required=True)
    parser.add_argument("--eval-flow", nargs="+", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--stride", type=int, default=200)
    parser.add_argument("--max-train-windows", type=int, default=None, help="Cap total train windows (None=all).")
    parser.add_argument("--max-eval-windows", type=int, default=None, help="Cap eval windows per sequence.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--disable-cudnn", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if len(args.train_h5) != len(args.train_flow):
        raise ValueError("--train-h5 and --train-flow must have the same number of entries.")
    if len(args.eval_h5) != len(args.eval_flow):
        raise ValueError("--eval-h5 and --eval-flow must have the same number of entries.")

    if args.disable_cudnn:
        import torch
        torch.backends.cudnn.enabled = False

    import torch
    import torch.nn.functional as F
    from mvsec_benchmark.models.evflownet_like import EVFlowNetLike

    torch.manual_seed(args.seed)

    adapters = build_adapters()
    if args.adapter not in adapters:
        raise KeyError(f"Unknown adapter: {args.adapter}. Available: {list(adapters)}")
    adapter = adapters[args.adapter]

    # --- Load and build representations for all training sequences ---
    print(f"[outdoor_suite] adapter={args.adapter}, device={args.device}")
    print(f"[outdoor_suite] loading {len(args.train_h5)} train sequence(s)...")

    train_reps: list[np.ndarray] = []
    train_flows: list[np.ndarray] = []

    for h5_path, flow_path in zip(args.train_h5, args.train_flow):
        print(f"  train: {h5_path.name}")
        samples = load_mvsec_windows(
            h5_path=h5_path,
            flow_path=flow_path,
            window_size=args.window_size,
            stride=args.stride,
            max_windows=args.max_train_windows,
        )
        for s in samples:
            train_reps.append(adapter.build(s.events, s.sensor_size))
            train_flows.append(np.moveaxis(s.gt_flow, -1, 0))
        print(f"    windows: {len(samples)}")

    print(f"[outdoor_suite] total train windows: {len(train_reps)}")

    x_train = torch.from_numpy(np.stack(train_reps, axis=0)).float().to(args.device)
    y_train = torch.from_numpy(np.stack(train_flows, axis=0)).float().to(args.device)
    in_channels = int(x_train.shape[1])

    # --- Train EV-FlowNet ---
    model = EVFlowNetLike(in_channels=in_channels, base_channels=args.base_channels).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_train = int(x_train.shape[0])

    print(f"[outdoor_suite] training {args.epochs} epochs on {n_train} windows (in_channels={in_channels})...")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=args.device)
        epoch_loss = 0.0
        for start in range(0, n_train, args.batch_size):
            idx = perm[start:start + args.batch_size]
            pred = model(x_train[idx])
            loss = F.smooth_l1_loss(pred, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{args.epochs}  loss={epoch_loss / n_train:.4f}")

    # --- Evaluate on each indoor sequence separately ---
    model.eval()
    seq_results: list[dict] = []
    eval_names = [h5.stem.replace("_left_events", "") for h5 in args.eval_h5]

    for seq_name, h5_path, flow_path in zip(eval_names, args.eval_h5, args.eval_flow):
        print(f"[outdoor_suite] eval: {seq_name}")
        samples = load_mvsec_windows(
            h5_path=h5_path,
            flow_path=flow_path,
            window_size=args.window_size,
            stride=args.stride,
            max_windows=args.max_eval_windows,
        )
        eval_reps = [adapter.build(s.events, s.sensor_size) for s in samples]
        x_eval = torch.from_numpy(np.stack(eval_reps, axis=0)).float().to(args.device)

        with torch.no_grad():
            preds = model(x_eval).cpu().numpy()

        metrics = [
            compute_flow_metrics(np.moveaxis(pred, 0, -1), s.gt_flow)
            for pred, s in zip(preds, samples)
        ]
        aee = float(np.mean([m.aee for m in metrics]))
        outlier = float(np.mean([m.outlier_percent for m in metrics]))
        valid = int(sum(m.valid_count for m in metrics))
        print(f"  {seq_name}: AEE={aee:.4f}  Outlier={outlier:.2f}%  windows={len(samples)}")
        seq_results.append({
            "sequence": seq_name,
            "aee": aee,
            "outlier_percent": outlier,
            "eval_windows": len(samples),
            "valid_count": valid,
        })

    # Average across sequences
    mean_aee = float(np.mean([r["aee"] for r in seq_results]))
    mean_outlier = float(np.mean([r["outlier_percent"] for r in seq_results]))

    result = {
        "adapter_name": args.adapter,
        "channels": in_channels,
        "train_sequences": [h5.name for h5 in args.train_h5],
        "eval_sequences": eval_names,
        "total_train_windows": len(train_reps),
        "epochs": args.epochs,
        "mean_aee": mean_aee,
        "mean_outlier_percent": mean_outlier,
        "per_sequence": seq_results,
    }

    payload = json.dumps(result, indent=2, sort_keys=True)
    print("\n" + payload)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
