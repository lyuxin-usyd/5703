# Photometric + Smoothness V9 MatrixLSTM-Style Full-Split Summary

Dataset: MVSEC `outdoor_day1 + outdoor_day2` for training and `indoor_flying1/2/3` for evaluation.

V9 defaults: EV-FlowNet-style multi-scale decoder, MatrixLSTM-style image-pair windows, raw 0-255 grayscale images, train image strides 1..5 sampled lazily per epoch, 256 crop, random flip/rotation augmentation, self-supervised image-pair photometric warping loss, and flow smoothness loss. Ground-truth flow is used only for validation and evaluation metrics.

Lower is better for AEE and Outlier %.

| Method | AEE | Outlier % | Best val AEE | Best epoch | Epochs | Train windows | Eval windows |
|---|---:|---:|---:|---:|---:|---:|---:|
| ERGO | 1.3463 | 7.72 | 0.7001 | 32 | 42 | 728 | 2589 |
| EST | 1.3895 | 8.41 | 0.6695 | 34 | 44 | 728 | 2589 |
| Event Pre-training | 1.7844 | 15.63 | 0.8493 | 33 | 43 | 728 | 2589 |
| EvRepSL | 1.7607 | 15.14 | 0.8001 | 36 | 46 | 728 | 2589 |
| GET | 1.4039 | 8.51 | 0.6972 | 33 | 43 | 728 | 2589 |
| MatrixLSTM | 1.4842 | 9.30 | 0.8109 | 8 | 18 | 728 | 2589 |

## Per-Sequence Breakdown

| Method | Sequence | AEE | Outlier % | Windows | Valid pixels | Outlier pixels |
|---|---|---:|---:|---:|---:|---:|
| ERGO | indoor_flying1 | 1.1435 | 4.46 | 945 | 3584763 | 220847 |
| ERGO | indoor_flying2 | 1.7588 | 14.98 | 710 | 3811246 | 1037645 |
| ERGO | indoor_flying3 | 1.2380 | 5.50 | 934 | 3672997 | 395482 |
| EST | indoor_flying1 | 1.2094 | 5.24 | 945 | 3584763 | 266728 |
| EST | indoor_flying2 | 1.7825 | 15.89 | 710 | 3811246 | 1103626 |
| EST | indoor_flying3 | 1.2729 | 5.92 | 934 | 3672997 | 429896 |
| Event Pre-training | indoor_flying1 | 1.5064 | 8.95 | 945 | 3584763 | 478941 |
| Event Pre-training | indoor_flying2 | 2.2994 | 26.24 | 710 | 3811246 | 1765431 |
| Event Pre-training | indoor_flying3 | 1.6741 | 14.33 | 934 | 3672997 | 963605 |
| EvRepSL | indoor_flying1 | 1.4341 | 8.21 | 945 | 3584763 | 416446 |
| EvRepSL | indoor_flying2 | 2.3448 | 28.28 | 710 | 3811246 | 1877143 |
| EvRepSL | indoor_flying3 | 1.6473 | 12.16 | 934 | 3672997 | 816948 |
| GET | indoor_flying1 | 1.2106 | 5.10 | 945 | 3584763 | 253823 |
| GET | indoor_flying2 | 1.8190 | 16.16 | 710 | 3811246 | 1115951 |
| GET | indoor_flying3 | 1.2839 | 6.15 | 934 | 3672997 | 445081 |
| MatrixLSTM | indoor_flying1 | 1.3000 | 5.85 | 945 | 3584763 | 289145 |
| MatrixLSTM | indoor_flying2 | 1.9001 | 17.52 | 710 | 3811246 | 1193120 |
| MatrixLSTM | indoor_flying3 | 1.3544 | 6.55 | 934 | 3672997 | 463275 |
