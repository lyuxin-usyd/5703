# MatrixLSTM Paper-Like Probe

This branch adds an isolated experiment for checking whether the gap to the
MatrixLSTM / EV-FlowNet optical-flow numbers mainly comes from protocol
differences.

It does not replace the formal six-method benchmark line.

Implemented in this probe:

- Dense GT flow propagation across the requested event interval, following the
  key evaluation idea in MatrixLSTM / EV-FlowNet.
- Event-mask AEE and KITTI-style outlier percentage: only pixels with events and
  non-zero valid GT flow are counted.
- A separate runner under `scripts/run_matrixlstm_paperlike_probe.py` and
  `scripts/run_matrixlstm_paperlike_probe.sh`.

Not implemented yet:

- The original TensorFlow EV-FlowNet architecture.
- Image-pair photometric warping loss.
- The original TFRecord image-pair loader.

AutoDL example:

```bash
cd /root/autodl-tmp/capstone/5703_eventvalid/optical-flow
git fetch origin
git switch codex/matrixlstm-paperlike-probe
DATA_ROOT=/root/autodl-tmp/capstone/data/mvsec \
OMP_NUM_THREADS=8 \
BATCH_SIZE=8 \
EPOCHS=100 \
bash scripts/run_matrixlstm_paperlike_probe.sh
```

To try another adapter while keeping the same paper-like protocol:

```bash
ADAPTER=est DATA_ROOT=/root/autodl-tmp/capstone/data/mvsec \
OMP_NUM_THREADS=8 BATCH_SIZE=8 bash scripts/run_matrixlstm_paperlike_probe.sh
```
