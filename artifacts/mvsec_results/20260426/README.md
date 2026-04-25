# MVSEC Original-Protocol Result Package, 2026-04-26

This folder tracks the lightweight result artifact for the optical-flow task.
It is safe to keep in Git because it contains only logs, JSON metric outputs,
and documentation snapshots.

It does not contain raw MVSEC data, extracted event HDF5 files, generated flow
NPZ files, model checkpoints, or any other large dataset artifact.

## Included Archive

```text
mvsec_original_protocol_results_20260426.tar.gz
```

## Protocol

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- events: 6M extracted left-camera events per sequence
- flow GT: generated full `*_gt_flow_full.npz`
- decoder: shared `EVFlowNetLike`
- epochs: 1
- GPU: AutoDL RTX 4090

## Result Summary

| Method | AEE | Outlier % |
|---|---:|---:|
| ERGO | 3.007065 | 38.588946 |
| EST | 2.935838 | 38.963377 |
| Event Pre-training | 2.948537 | 38.059867 |
| EvRepSL | 3.032907 | 39.112478 |
| GET | 3.016356 | 38.631755 |
| MatrixLSTM | 3.059037 | 39.388230 |

Interpretation: this is a unified adapted MVSEC optical-flow reproduction
benchmark. It is not a paper-identical rerun of every method's original private
training stack.
