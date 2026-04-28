# MVSEC Optical Flow Task Review, 2026-04-26

This is the current source of truth for the optical-flow part of the COMP5703
benchmark work.

## What Is Complete

The code now has a closed runnable path for six methods:

- `est`
- `ergo`
- `event_pretraining`
- `evrepsl`
- `get`
- `matrixlstm`

`omnievent` remains reported-only because the current public code path does not
provide the needed runnable optical-flow downstream implementation.

The formal MVSEC protocol has been run on AutoDL:

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event cap: 6M events per sequence HDF5
- flow GT: full generated `*_gt_flow_full.npz`
- decoder: shared `EVFlowNetLike`
- completed run: epochs 1, used as baseline/sanity check
- recommended formal budget: max epochs 100 with outdoor-validation early stopping
- GPU: RTX 4090

Result archive:

```text
results/autodl_archives/20260426/mvsec_original_protocol_results_20260426.tar.gz
```

Extracted result JSONs:

```text
results/autodl_archives/20260426/extracted/results/
```

## Formal Result Table

| Method | AEE | Outlier % | Train windows | Eval windows | Valid count |
|---|---:|---:|---:|---:|---:|
| ERGO | 3.007065 | 38.588946 | 17329 | 3583 | 322326680 |
| EST | 2.935838 | 38.963377 | 17329 | 3583 | 322326680 |
| Event Pre-training | 2.948537 | 38.059867 | 17329 | 3583 | 322326680 |
| EvRepSL | 3.032907 | 39.112478 | 17329 | 3583 | 322326680 |
| GET | 3.016356 | 38.631755 | 17329 | 3583 | 322326680 |
| MatrixLSTM | 3.059037 | 39.388230 | 17329 | 3583 | 322326680 |

These are the main reproduction/adaptation baseline results currently available. The
older `smoke`, `indoor_100f_2m`, and `indoor_full_6m` archives are debugging
and controlled experiments, not the main result.

## Recommended Formal Training Budget

The `epochs=1` result is a runnable baseline, not a convergence result. For the
formal adapted MVSEC run, use:

- `--epochs 100`
- `--early-stop-val-windows 1000`
- `--early-stop-patience 10`
- `--early-stop-min-delta 0.001`
- `--progress-every 100`

Early stopping uses held-out outdoor training windows only. The indoor flying
sequences remain final evaluation data and are not used to decide when training
stops.

Use `scripts/run_mvsec_100e_all_early_stop.sh` on AutoDL to run all six methods
sequentially with logs and progress output.

## Closed-Loop Check

The engineering loop is closed:

```text
MVSEC bag files
  -> event HDF5 extraction
  -> GT flow NPZ generation
  -> six event representation adapters
  -> shared optical-flow decoder
  -> AEE / outlier metrics
  -> archived logs and JSON outputs
```

The reporting loop is also mostly closed:

```text
paper scope
  -> runnable/non-runnable classification
  -> protocol choice
  -> experiment evidence
  -> limitations for final report
```

The scientific reproduction loop is not fully paper-faithful yet. The current
result should be described as a unified adapted reproduction benchmark, not as
exactly matching each paper's private optical-flow training stack.

## Review Findings

### P1: The current event-window / flow pairing is index based, not timestamp based

File: `src/mvsec_benchmark/data/mvsec.py`

`iter_event_windows()` slices fixed event-count windows and pairs the first
event window with the first flow frame, the second with the second flow frame,
and so on. This prevents the old repeated-flow bug, but it is still not the
paper-faithful MVSEC pairing where events should be matched to the exact time
interval around each flow timestamp or image frame.

Impact:

- The benchmark is internally consistent across all six methods.
- The numbers should not be claimed as exact paper reproduction numbers.
- For the final report, say: "We reproduce the MVSEC train/eval split with a
  unified adapted event-window protocol."

### P1: The decoder is shared and simplified

File: `src/mvsec_benchmark/models/evflownet_like.py`

All methods use the same lightweight EV-FlowNet-like decoder. This is good for
fair representation comparison, but it is not the same as each paper's original
private downstream head or full training recipe.

Impact:

- This is valid for a benchmark study.
- It is not valid to directly compare the raw AEE numbers against paper tables
  without explaining the decoder difference.

### P2: One epoch is a minimal reproduction run, not convergence training

File: `scripts/run_original_protocol.py`

The completed formal-style run used `--epochs 1`. That was enough to prove the
protocol can run end to end and produce stable logs/results, but it is not a
convergence study.

Impact:

- Good enough as an initial reproducible baseline.
- For stronger numeric claims, run the same protocol with a 100-epoch maximum
  and outdoor-validation early stopping.

### P2: Old outdoor helper script was misleading

File: `scripts/autodl_outdoor_pipeline.sh`

The old version referenced stale filenames such as `*_left_events.h5` and
`*_gt_flow.npz`, and called the older `run_outdoor_suite.py`. It has now been
replaced with the successful `run_original_protocol.py` command using
`*_left_events_6m.h5` and `*_gt_flow_full.npz`.

Impact:

- Future AutoDL runs should use `bash scripts/autodl_outdoor_pipeline.sh`.
- Do not use `run_outdoor_suite.py` for the final report path unless explicitly
  debugging legacy behavior.

### P3: Progress logs can look noisy

File: `src/mvsec_benchmark/pipeline.py`

Training samples are shuffled, so "built representation X/total" messages do
not appear monotonically. This is not a correctness issue, but it confused the
run monitoring.

Impact:

- If this becomes annoying, change the log to count processed batches rather
  than original sample indices.

## What To Say In The Report

Use this wording:

> We implemented a unified MVSEC optical-flow benchmark for six runnable event
> representation methods. For methods with no released optical-flow downstream
> code, we added an adapted representation-to-flow path. We used the common
> MVSEC split of outdoor_day1/2 for training and indoor_flying1/2/3 for
> evaluation, and compared all methods under the same shared EV-FlowNet-like
> decoder and metrics.

Do not say:

> We exactly reproduced all paper numbers.

Better wording:

> The reproduced numbers are protocol-aligned but not paper-identical, because
> several papers do not release their optical-flow training code and we use a
> unified decoder for fair comparison.

## Next Recommended Step

For the current submission, the recommended next experiment is the 100-epoch
maximum early-stop run:

1. Run `scripts/run_mvsec_100e_all_early_stop.sh` on AutoDL.
2. Replace or supplement the baseline table with the resulting `e100_earlystop`
   JSON files.
3. Explain the limitations clearly: index-based pairing and shared decoder.
