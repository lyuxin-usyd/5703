# Claude Code Handoff: MVSEC Optical Flow Benchmark

Last updated: 2026-04-26

This file is the machine-readable handoff for continuing the COMP5703 MVSEC
optical-flow reproduction work. Do not rely only on the Obsidian note; this
document is the operational source of truth for Claude Code.

## Current Goal

We are reproducing / adapting six runnable event-representation papers on the
MVSEC optical-flow downstream task. OmniEvent is reported-only and should not be
rerun.

The final protocol we are trying to run is the original-style MVSEC split:

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
experiments.

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

## Current Next Experiment

Run the formal protocol for one adapter first. Use `est` first because it is
the clearest baseline.

This trains on outdoor day 1/2 and evaluates on all three indoor flying
sequences. It does not rerun the old indoor-only benchmark.

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

## After EST Succeeds

Run the same command for the other adapters by replacing `--adapter` and the
output/log names:

```text
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

