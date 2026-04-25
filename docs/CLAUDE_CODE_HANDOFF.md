# Claude Code Handoff: MVSEC Optical Flow Benchmark

Last updated: 2026-04-26

This file is the machine-readable handoff for continuing the COMP5703 MVSEC
optical-flow reproduction work. Do not rely only on the Obsidian note; this
document is the operational source of truth for Claude Code.

## Current Goal

We are reproducing / adapting six runnable event-representation papers on the
MVSEC optical-flow downstream task. OmniEvent is reported-only and should not be
rerun.

The final protocol has completed on the original-style MVSEC split:

- train: `outdoor_day1 + outdoor_day2`
- evaluate: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- metric: `AEE` and `outlier_percent`
- shared decoder: `src/mvsec_benchmark/models/evflownet_like.py`
- runnable adapters: `est`, `ergo`, `event_pretraining`, `get`, `matrixlstm`,
  `evrepsl`

Important wording: when this protocol mentions `indoor_flying1/2/3`, that is
the evaluation set. It is not the already-finished indoor-only experiment.

## Repository State

Local repository:

```text
D:\event-benchmark\mvsec-benchmark
```

GitHub repository:

```text
https://github.com/lyuxin-usyd/5703
```

Branch:

```text
codex/mvsec-runnable
```

Important commits already pushed:

```text
28b06a3 Add outdoor train/indoor eval benchmark pipeline
f82039d Allow full MVSEC flow generation
06292b9 Stream original protocol representations
1b5d719 Add original MVSEC train eval protocol runner
```

If AutoDL is on an old copy, refresh the code from GitHub before running more
experiments. The current helper script is `scripts/autodl_outdoor_pipeline.sh`;
older folders such as `5703_before_output_fix` and `5703_before_stream_fix_*`
are archived debugging copies.

## Completed Experiments

These are already done. Do not repeat them unless explicitly debugging.

### 1. Smoke / tiny real-data checks

Archive:

```text
mvsec_smoke_results_20260425.tar.gz
```

### 2. Indoor controlled benchmark, 100 GT frames + 2M events

Archive:

```text
mvsec_indoor_100f_2m_results_20260425.tar.gz
```

Mean results:

```text
ergo              AEE=0.578881 outlier=2.729658
est               AEE=0.589735 outlier=2.744183
event_pretraining AEE=0.565045 outlier=2.768675
evrepsl           AEE=0.619531 outlier=2.716874
get               AEE=0.561336 outlier=2.731251
matrixlstm        AEE=0.613501 outlier=2.721506
```

### 3. Indoor controlled benchmark, full indoor GT + 6M events

Archive:

```text
mvsec_indoor_full_6m_results_20260425.tar.gz
```

Mean results:

```text
ergo              AEE=2.573404 outlier=31.738944
est               AEE=2.569132 outlier=31.676690
event_pretraining AEE=2.587144 outlier=31.895095
evrepsl           AEE=2.577776 outlier=31.724076
get               AEE=2.576629 outlier=31.699674
matrixlstm        AEE=2.578021 outlier=31.715153
```

### 4. Formal original-style protocol, outdoor train + indoor eval

Archive:

```text
mvsec_original_protocol_results_20260426.tar.gz
```

Local copies:

```text
D:\event-benchmark\mvsec-benchmark\results\autodl_archives\20260426
D:\ObsidianVault\02-项目记录\COMP5703-Capstone\artifacts\mvsec_results\20260426
```

Protocol:

```text
train: outdoor_day1 + outdoor_day2
eval:  indoor_flying1 + indoor_flying2 + indoor_flying3
events: 6M events per sequence HDF5
flow:   full generated *_gt_flow_full.npz
decoder: shared EVFlowNetLike
epochs: 1
GPU: RTX 4090
```

Formal results:

```text
ergo              AEE=3.007065 outlier=38.588946 train=17329 eval=3583
est               AEE=2.935838 outlier=38.963377 train=17329 eval=3583
event_pretraining AEE=2.948537 outlier=38.059867 train=17329 eval=3583
evrepsl           AEE=3.032907 outlier=39.112478 train=17329 eval=3583
get               AEE=3.016356 outlier=38.631755 train=17329 eval=3583
matrixlstm        AEE=3.059037 outlier=39.388230 train=17329 eval=3583
```

Important interpretation:

- This is the current main result for the optical-flow task.
- It is an adapted unified benchmark, not an exact reproduction of each
  paper's private downstream training stack.
- The current code pairs fixed event-count windows with flow frames by index.
  This is internally consistent across methods but not the most paper-faithful
  timestamp-aligned MVSEC pairing.

These archive files are also copied locally under:

```text
D:\event-benchmark\mvsec-benchmark\results\autodl_archives\20260425
D:\ObsidianVault\02-项目记录\COMP5703-Capstone\artifacts\mvsec_results\20260425
```

## AutoDL Data State

Expected AutoDL root:

```text
/root/autodl-tmp/capstone
```

Expected repository path:

```text
/root/autodl-tmp/capstone/5703
```

Expected data path:

```text
/root/autodl-tmp/capstone/data/mvsec
```

Known data files from the latest run:

```text
/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day1_data.bag
/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day1_gt.bag
/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day2_data.bag
/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day2_gt.bag
/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day1_gt_flow_full.npz
/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day2_gt_flow_full.npz
/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day1_left_events_6m.h5
/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day2_left_events_6m.h5
/root/autodl-tmp/capstone/data/mvsec/indoor_flying1/indoor_flying1_left_events_6m.h5
/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying1_gt_flow_full.npz
/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying2_left_events_6m.h5
/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying2_gt_flow_full.npz
/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying3_left_events_6m.h5
/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying3_gt_flow_full.npz
```

Path warning:

- `indoor_flying1_left_events_6m.h5` is under `indoor_flying1/`.
- `indoor_flying2/3` event files and all indoor `*_gt_flow_full.npz` files are
  under `indoor_flying/`.
- A common failure is using
  `/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying1_left_events_6m.h5`,
  which does not exist.

## First Check On AutoDL

Run this before any long experiment:

```bash
cd /root/autodl-tmp/capstone/5703
pwd
ls
git log --oneline -5 2>/dev/null || true
grep -n "progress-every\|total_windows" scripts/run_original_protocol.py
grep -n "flow_limit\|n_yielded >= flow_limit" src/mvsec_benchmark/data/mvsec.py
```

Expected signs:

```text
/root/autodl-tmp/capstone/5703
README.md configs docs pyproject.toml scripts src tests
--progress-every
total_windows
flow_limit
n_yielded >= flow_limit
```

If `cd /root/autodl-tmp/capstone/5703` fails, the user is in the wrong folder.
Do not run from `/root/autodl-tmp/capstone`.

## Re-run Formal Protocol If Needed

The formal protocol has already completed for all six runnable adapters. Do not
repeat it unless debugging or intentionally producing a longer training run.

If it must be rerun, the helper script now points to the current successful
protocol:

```bash
cd /root/autodl-tmp/capstone/5703
bash scripts/autodl_outdoor_pipeline.sh
```

For one adapter only:

```bash
cd /root/autodl-tmp/capstone/5703
bash scripts/autodl_outdoor_pipeline.sh est
```

The raw command below is kept for reference. It trains on outdoor day 1/2 and
evaluates on all three indoor flying sequences. It does not rerun the old
indoor-only benchmark.

```bash
cd /root/autodl-tmp/capstone/5703
mkdir -p logs results

python scripts/run_original_protocol.py \
  --adapter est \
  --train-pair /root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day1_left_events_6m.h5:/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day1_gt_flow_full.npz \
  --train-pair /root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day2_left_events_6m.h5:/root/autodl-tmp/capstone/data/mvsec/outdoor_day/outdoor_day2_gt_flow_full.npz \
  --eval-pair /root/autodl-tmp/capstone/data/mvsec/indoor_flying1/indoor_flying1_left_events_6m.h5:/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying1_gt_flow_full.npz \
  --eval-pair /root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying2_left_events_6m.h5:/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying2_gt_flow_full.npz \
  --eval-pair /root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying3_left_events_6m.h5:/root/autodl-tmp/capstone/data/mvsec/indoor_flying/indoor_flying3_gt_flow_full.npz \
  --epochs 1 \
  --batch-size 4 \
  --eval-batch-size 1 \
  --device cuda \
  --disable-cudnn \
  --progress-every 100 \
  --output results/original_od12_if123_full6m_est_e1_stream.json \
  2>&1 | tee logs/original_od12_if123_full6m_est_e1_stream.log
```

Expected progress output should include:

```text
[load:train] pair 1/2 ...
[load:train] pair 1/2 windows=...
[load:train] pair 2/2 windows=...
[load:train] total_windows=...
[load:eval] pair 1/3 ...
[load:eval] total_windows=...
[setup] adapter=est train_windows=...
[train] epoch 1/1 batches=...
[eval] batch ...
```

If there is no progress for many minutes, do not assume it is fine. Check that
the code has the progress patch and that the terminal is actually in
`/root/autodl-tmp/capstone/5703`.

## Adapter List

The formal run should include exactly these six runnable adapters:

```text
est
ergo
event_pretraining
get
matrixlstm
evrepsl
```

Do this sequentially, not in parallel, on one 4090. Parallel runs can fight for
VRAM and make diagnosis harder.

## Archive Results

After a successful run:

```bash
cd /root/autodl-tmp/capstone/5703
tar -czf /root/autodl-tmp/capstone/mvsec_original_protocol_results_$(date +%Y%m%d).tar.gz results logs docs README.md
ls -lh /root/autodl-tmp/capstone/mvsec_original_protocol_results_*.tar.gz
```

Download the archive from AutoDL Jupyter file browser. It is small because it
contains JSON/logs, not raw MVSEC bags.

## Known Pitfalls

1. Do not rerun indoor-only controlled benchmarks unless debugging. They are
   already archived.
2. Do not call the formal protocol "outdoor only". It is outdoor-train and
   indoor-test.
3. Do not use 100-frame GT flow for final original-style protocol. The full
   flow files are named `*_gt_flow_full.npz`.
4. Do not run commands from `/root/autodl-tmp/capstone`; run from
   `/root/autodl-tmp/capstone/5703`.
5. Do not expect exact paper numbers. Several official repos do not release the
   optical-flow downstream code, so this is a unified adapted reproduction.
6. If AutoDL Jupyter terminal freezes, prefer a fresh terminal and inspect
   `logs/*.log`. For long runs, use `tee` so progress is preserved.
7. Do not use `scripts/run_outdoor_suite.py` for the final protocol unless
   explicitly debugging legacy behavior. Use `scripts/run_original_protocol.py`
   or `scripts/autodl_outdoor_pipeline.sh`.
