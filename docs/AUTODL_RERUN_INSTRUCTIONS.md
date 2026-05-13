# AutoDL MVSEC Optical-Flow Rerun Instructions

This is the handoff checklist for rerunning the optical-flow benchmark on AutoDL.

## 1. Update Code

```bash
cd /root/autodl-tmp/capstone/5703
git pull
```

The updated runner uses:

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- methods: `ergo`, `est`, `event_pretraining`, `evrepsl`, `get`, `matrixlstm`
- shared decoder: `EVFlowNetLike`
- max epochs: 100
- batch size: 8
- early-stop patience: 10
- validation: block-random outdoor validation
- event/flow pairing: timestamp-aligned event intervals from flow GT timestamps

## 2. Check indoor_flying1 GT

Use the corrected `indoor_flying1` GT file before rerunning:

```bash
cd /root/autodl-tmp/capstone/data/mvsec/indoor_flying
cp -av indoor_flying1_gt_flow_2000.npz indoor_flying1_gt_flow_full.npz

python - <<'PY'
import numpy as np
f = "/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying1_gt_flow_full.npz"
with np.load(f) as d:
    print(d["x_flow_dist"].shape)
    print(len(d["timestamps"]))
PY
```

Expected output:

```text
(1398, 260, 346)
1398
```

## 3. Run the Six-Method Benchmark

```bash
cd /root/autodl-tmp/capstone/5703
OMP_NUM_THREADS=8 BATCH_SIZE=8 bash scripts/run_mvsec_100e_all_early_stop.sh
```

The runner writes:

- per-method JSON results under `results/`
- per-method logs under `logs/`
- per-method curve CSV files under `logs/curves/`

## 4. Build Tables and Figures

After the benchmark finishes:

```bash
python scripts/build_mvsec_e100_outputs.py \
  --results-dir results \
  --curve-dir logs/curves \
  --summary-dir results/summary \
  --figures-dir results/figures
```

Expected generated deliverables:

- `results/summary/mvsec_e100_earlystop_summary.csv`
- `results/summary/mvsec_e100_earlystop_summary.md`
- `results/figures/mvsec_e100_earlystop_aee.svg`
- `results/figures/mvsec_e100_earlystop_outlier.svg`
- `results/figures/mvsec_e100_earlystop_train_loss_curve.svg`
- `results/figures/mvsec_e100_earlystop_val_curve.svg`

## 5. Package the Rerun

```bash
tar -czf /root/autodl-tmp/capstone/mvsec_timestamp_bs8_results_$(date +%Y%m%d_%H%M).tar.gz \
  results logs docs README.md README_FOR_GROUP.md
```

Send back the generated `.tar.gz` package or push the updated `results/`,
`logs/curves/`, and summary files after checking the run completed.

