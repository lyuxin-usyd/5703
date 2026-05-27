# Photometric + Smoothness V9 MatrixLSTM-Style Full-Split Summary

Dataset: MVSEC `outdoor_day1 + outdoor_day2` for training and `indoor_flying1/2/3` for evaluation.

V9 defaults: EV-FlowNet-style multi-scale decoder, MatrixLSTM-style image-pair windows, raw 0-255 grayscale images, train image strides 1..5 sampled lazily per epoch, 256 crop, random flip/rotation augmentation, self-supervised image-pair photometric warping loss, and flow smoothness loss. Ground-truth flow is used only for validation and evaluation metrics.

Lower is better for AEE and Outlier %.

| Method | AEE | Outlier % | Best val AEE | Best epoch | Epochs | Train windows | Eval windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| Event Frame | 1.8615 | 16.79 | 0.9268 | 24 | 34 | 728 | 2589 |
| Binary Event Image | 1.6993 | 14.68 | 0.6645 | 44 | 54 | 728 | 2589 |
| Timestamp Image | 1.8271 | 15.25 | 0.9627 | 24 | 34 | 728 | 2589 |
| Time Surface | 1.7234 | 15.17 | 0.7380 | 24 | 34 | 728 | 2589 |
| Voxel Grid | 1.5476 | 10.91 | 0.8996 | 29 | 39 | 728 | 2589 |

## Per-Sequence Breakdown

| Method | Sequence | AEE | Outlier % | Windows | Valid pixels | Outlier pixels |
|---|---|---:|---:|---:|---:|---:|
| Event Frame | indoor_flying1 | 1.5704 | 10.52 | 945 | 3584763 | 568618 |
| Event Frame | indoor_flying2 | 2.4024 | 26.53 | 710 | 3811246 | 1789895 |
| Event Frame | indoor_flying3 | 1.7448 | 15.74 | 934 | 3672997 | 1038468 |
| Binary Event Image | indoor_flying1 | 1.4075 | 8.28 | 945 | 3584763 | 409438 |
| Binary Event Image | indoor_flying2 | 2.2357 | 25.51 | 710 | 3811246 | 1718067 |
| Binary Event Image | indoor_flying3 | 1.5867 | 12.90 | 934 | 3672997 | 855839 |
| Timestamp Image | indoor_flying1 | 1.6049 | 9.69 | 945 | 3584763 | 504276 |
| Timestamp Image | indoor_flying2 | 2.2646 | 24.71 | 710 | 3811246 | 1631941 |
| Timestamp Image | indoor_flying3 | 1.7194 | 13.70 | 934 | 3672997 | 901054 |
| Time Surface | indoor_flying1 | 1.4415 | 8.51 | 945 | 3584763 | 419129 |
| Time Surface | indoor_flying2 | 2.2562 | 25.68 | 710 | 3811246 | 1715945 |
| Time Surface | indoor_flying3 | 1.6035 | 13.91 | 934 | 3672997 | 906995 |
| Voxel Grid | indoor_flying1 | 1.2859 | 6.65 | 945 | 3584763 | 346630 |
| Voxel Grid | indoor_flying2 | 2.0415 | 20.97 | 710 | 3811246 | 1427480 |
| Voxel Grid | indoor_flying3 | 1.4370 | 7.58 | 934 | 3672997 | 536857 |
