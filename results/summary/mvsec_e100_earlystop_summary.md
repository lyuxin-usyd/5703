# MVSEC Optical Flow E100 Early-Stop Summary

Protocol: train on `outdoor_day1 + outdoor_day2`, evaluate on `indoor_flying1/2/3`, event window 6M, full generated GT flow frames, max 100 epochs, early-stop patience 10, block-random validation from outdoor train set.

Lower is better for AEE and outlier percentage.

| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE | Train windows | Eval windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ergo | 2.9713 | 38.31 | 61.69 | 20 | 10 | 2.6049 | 16329 | 3583 |
| est | 2.8654 | 37.04 | 62.96 | 11 | 1 | 2.6380 | 16329 | 3583 |
| event_pretraining | 2.9653 | 38.19 | 61.81 | 20 | 10 | 2.6004 | 16329 | 3583 |
| evrepsl | 3.0180 | 39.06 | 60.94 | 15 | 5 | 2.6243 | 16329 | 3583 |
| get | 2.9619 | 38.34 | 61.66 | 22 | 12 | 2.6007 | 16329 | 3583 |
| matrixlstm | 3.0138 | 38.97 | 61.03 | 22 | 12 | 2.6071 | 16329 | 3583 |
| OmniEvent✳ | 0.9900 | 3.24 | 96.76 | paper | paper | paper | paper | paper |

✳ OmniEvent is a paper-reported reference row. Its displayed values are simple averages from the OmniEvent paper's MVSEC `indoor_flying1/2/3` results, not a local run in this repository. Source: [arXiv:2508.01842](https://arxiv.org/abs/2508.01842), Table 2.

## Reporting Notes

- This is a unified downstream optical-flow benchmark / adapted reproduction, not an exact reimplementation of each paper's original optical-flow decoder.
- The six event representations are compared under the same EVFlowNet-like decoder and the same MVSEC train/eval protocol, so the comparison is fair inside this benchmark.
- The runner has been updated for the next rerun to pair event windows to flow frames by timestamp intervals when flow timestamps are available. The table above is the checked-in earlier result package.
- `OmniEvent✳` is included only as reported-only context and should not be ranked as a local result from this pipeline.
- W&B hooks exist in the code. The current deliverable uses local CSV curves and SVG figures, which are enough for reporting unless an online W&B dashboard is explicitly required.
