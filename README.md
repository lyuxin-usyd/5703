# MVSEC Optical Flow Benchmark

This folder contains the optical-flow part of the COMP5703 benchmark group
work. The current formal optical-flow package is V9.

The repository keeps two V9 result parts:

```text
results/
  learning-base/   six learned representation methods
  baseline/        five traditional baseline representations
```

Both parts use the same MVSEC split and the same V9 photometric/smoothness
training and evaluation protocol.

## Current Protocol

- train: `outdoor_day1 + outdoor_day2`
- evaluate: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- event input: 6M extracted left-camera events per sequence
- image input: aligned grayscale image H5 files
- model: EV-FlowNet-style multi-scale decoder
- objective: image-pair photometric warping loss + flow smoothness loss
- training: max 100 epochs, batch size 8, early-stop patience 10
- validation: block-random outdoor validation
- train windows: 728
- eval windows: 2589
- metrics: AEE and KITTI-style Outlier over valid GT flow pixels

The method code is kept here:

```text
scripts/
src/mvsec_benchmark/
configs/envs/
```

## Result Layout

```text
results/learning-base/
  summary.csv
  summary.md
  per_sequence_summary.csv
  validation_aee_curves.png
  best_validation_aee_curves.png
  curves/
  results/
  artifacts/

results/baseline/
  summary.csv
  summary.md
  per_sequence_summary.csv
  validation_aee_curves.png
  best_validation_aee_curves.png
  curves/
  results/
  artifacts/

results/figures/
  mvsec_flow_inference_grid.png
  mvsec_flow_error_grid.png
```

## Latest Results

Learning-base methods:

| Method | AEE | Outlier % |
| --- | ---: | ---: |
| ERGO | 1.3463 | 7.72 |
| EST | 1.3895 | 8.41 |
| GET | 1.4039 | 8.51 |
| MatrixLSTM | 1.4842 | 9.30 |
| EvRepSL | 1.7607 | 15.14 |
| Event Pre-training | 1.7844 | 15.63 |

Traditional baseline methods under the same V9 protocol:

| Method | AEE | Outlier % |
| --- | ---: | ---: |
| Voxel Grid | 1.5476 | 10.91 |
| Binary Event Image | 1.6993 | 14.68 |
| Time Surface | 1.7234 | 15.17 |
| Timestamp Image | 1.8271 | 15.25 |
| Event Frame | 1.8615 | 16.79 |

## Code Layout

```text
optical-flow/
├── configs/
├── docs/
├── results/
│   ├── learning-base/
│   └── baseline/
├── scripts/
├── src/
│   └── mvsec_benchmark/
└── tests/
```

## Rerun Notes

Raw MVSEC bags and processed `.h5` / `.npz` files are not committed to GitHub.
Before rerunning, place processed MVSEC files outside the repository and pass
their location through `DATA_ROOT`.

Main V9 runner:

```bash
DATA_ROOT=/path/to/mvsec_full_processed \
DEVICE=cuda \
bash scripts/run_photometric_full_6methods_e100.sh
```

Traditional baseline representations can be run with the same V9 runner by
passing method names explicitly:

```bash
bash scripts/run_photometric_full_6methods_e100.sh \
  event_frame binary_event_image timestamp_image time_surface voxel_grid
```

## Reporting Notes

- Use `results/learning-base/summary.csv` for the main optical-flow table.
- Use `results/baseline/summary.csv` as the V9 baseline comparison table.
- Use `results/*/artifacts/<method>/` for checkpoints, fixed-sample inference
  outputs, manifests, and per-method metadata.
- Do not present these numbers as official paper reproduction numbers; this is
  a unified V9 downstream benchmark using a shared decoder and protocol.
