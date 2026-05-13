# MVSEC Optical Flow Project Status

Last updated: 2026-05-02

This artifact snapshot corresponds to the completed E100 early-stop MVSEC
optical-flow run. The current main result is the 100-epoch-maximum run with
early stopping.

## Current Main Result

- train: `outdoor_day1 + outdoor_day2`
- eval: `indoor_flying1 + indoor_flying2 + indoor_flying3`
- decoder: shared `EVFlowNetLike`
- max epochs: 100
- early stopping: patience 10
- validation: block-random outdoor validation
- metrics: AEE/EPE and KITTI-style outlier percentage
- train windows: 16329
- eval windows: 3583

| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE |
|---|---:|---:|---:|---:|---:|---:|
| ergo | 2.9713 | 38.31 | 61.69 | 20 | 10 | 2.6049 |
| est | 2.8654 | 37.04 | 62.96 | 11 | 1 | 2.6380 |
| event_pretraining | 2.9653 | 38.19 | 61.81 | 20 | 10 | 2.6004 |
| evrepsl | 3.0180 | 39.06 | 60.94 | 15 | 5 | 2.6243 |
| get | 2.9619 | 38.34 | 61.66 | 22 | 12 | 2.6007 |
| matrixlstm | 3.0138 | 38.97 | 61.03 | 22 | 12 | 2.6071 |
| OmniEvent✳ | 0.9900 | 3.24 | 96.76 | paper | paper | paper |

✳ OmniEvent is paper-reported only and was not reproduced locally.

## Interpretation

This should be described as a unified downstream optical-flow benchmark /
adapted reproduction. The six local methods use the same MVSEC split and the
same EVFlowNet-like downstream decoder.

Important limits:

- This is not an exact reimplementation of each paper's original optical-flow
  decoder/head.
- This artifact folder preserves the earlier checked-in result package. The
  active runner now uses timestamp-aligned event intervals when flow timestamps
  are available.
- `OmniEvent✳` is a reference row from the paper and should not be mixed with
  the six local runs as the same type of result.

## Artifact Contents

- `../results/*.json`: final local result JSONs
- `../logs/*.log`: training logs
- `../logs/curves/*.csv`: local training-loss and validation-AEE curves

Use the E100 early-stop result package as the current formal result.
