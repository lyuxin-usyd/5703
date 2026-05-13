# MVSEC Optical Flow E100 Early-Stop 汇报说明

## 实验定位

本实验是 COMP5703 benchmark 组中 optical-flow 方向的一版 MVSEC 统一下游 benchmark。它比较的是六个 runnable event representation 在同一个下游光流解码器上的表现。

这版结果应表述为 **unified downstream optical-flow benchmark / adapted reproduction**。不要表述为逐篇论文原始 optical-flow decoder/head 的完全复刻。

## 实验协议

- 数据集：MVSEC optical-flow sequences
- 训练集：`outdoor_day1 + outdoor_day2`
- 测试集：`indoor_flying1/2/3`
- 方法：`ergo`、`est`、`event_pretraining`、`evrepsl`、`get`、`matrixlstm`
- OmniEvent：不跑光流，按 reported-only 处理
- Event 输入：每个 sequence 抽取 6M left-camera events
- Flow GT：使用 full generated GT flow frames
- 下游网络：统一 `EVFlowNetLike` optical-flow decoder
- 训练上限：100 epochs
- Batch size：8
- Early stopping：patience 10
- Validation：从 outdoor training windows 中做 block-random validation
- Event-flow 配对：使用 flow GT timestamps 构造 timestamp-aligned event intervals
- 指标：AEE/EPE 和 Outlier %

Validation 不从 indoor test 集抽取，避免 test leakage。

## 主结果

Lower is better for AEE and Outlier %.

| Method | AEE | Outlier % | Non-outlier % | Epochs | Best epoch | Best val AEE | Train windows | Eval windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ergo | 2.9713 | 38.31 | 61.69 | 20 | 10 | 2.6049 | 16329 | 3583 |
| est | 2.8654 | 37.04 | 62.96 | 11 | 1 | 2.6380 | 16329 | 3583 |
| event_pretraining | 2.9653 | 38.19 | 61.81 | 20 | 10 | 2.6004 | 16329 | 3583 |
| evrepsl | 3.0180 | 39.06 | 60.94 | 15 | 5 | 2.6243 | 16329 | 3583 |
| get | 2.9619 | 38.34 | 61.66 | 22 | 12 | 2.6007 | 16329 | 3583 |
| matrixlstm | 3.0138 | 38.97 | 61.03 | 22 | 12 | 2.6071 | 16329 | 3583 |
| OmniEvent✳ | 0.9900 | 3.24 | 96.76 | paper | paper | paper | paper | paper |

在六个本地 runnable 方法里，当前最佳测试 AEE 是 EST，AEE=2.8654，Outlier=37.04%。但六个方法差距不大，汇报时更适合强调统一协议下的可比较性，而不是过度解读排名。

✳ OmniEvent 是 paper-reported 参考行，不是本仓库同一套 pipeline 的本地运行结果。表中的 AEE 和 Outlier 是 OmniEvent 论文 Table 2 在 `indoor_flying1/2/3` 上的简单平均值。来源：[arXiv:2508.01842](https://arxiv.org/abs/2508.01842)。

## 结果文件

- 结果表：`results/summary/mvsec_e100_earlystop_summary.csv`
- Markdown 结果表：`results/summary/mvsec_e100_earlystop_summary.md`
- AEE 柱状图：`results/figures/mvsec_e100_earlystop_aee.svg`
- Outlier 柱状图：`results/figures/mvsec_e100_earlystop_outlier.svg`
- Train loss 曲线：`results/figures/mvsec_e100_earlystop_train_loss_curve.svg`
- Validation AEE 曲线：`results/figures/mvsec_e100_earlystop_val_curve.svg`
- 原始 JSON/log/CSV 曲线：`artifacts/e100_earlystop_20260501/`

## 汇报口径

可以这样说：

> 我这部分做的是 MVSEC optical flow 的统一下游 benchmark。六个 runnable event representation 都接到同一个 EVFlowNet-like optical-flow decoder 上，在同一个训练/测试协议下比较。训练用 outdoor_day1/day2，测试用 indoor_flying1/2/3，最多 100 epoch，并用 outdoor train 内部的 block-random validation 做 early stopping。这样可以避免 indoor test 泄漏，也能保证六个 representation 的比较公平。

需要主动说明的限制：

- 当前不是逐篇论文原始 optical-flow decoder/head 的完全复刻，而是 adapted reproduction。
- Event 与 flow 的配对已经改为基于 flow GT timestamps 的时间区间配对，更接近 MVSEC optical-flow 任务口径。
- 结果可以支持组内 benchmark 对比，但不应直接声称超过或等同原论文 official numbers。
- `OmniEvent✳` 只能作为论文公开结果参考，不应和六个本地方法混成同一实验排名。
- W&B 支持已经在代码里，但本次主结果主要使用本地 CSV/SVG 曲线。

## 后续增强项

这些属于增强项，不影响当前主实验作为 adapted reproduction benchmark 提交：

- 原论文 decoder/head 的逐篇复刻
- 多 seed 重复实验
- 如 tutor 要求在线 dashboard，再打开 `--wandb-project` 补跑或导入 CSV
