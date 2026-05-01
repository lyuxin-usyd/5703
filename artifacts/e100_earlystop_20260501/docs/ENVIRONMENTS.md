# Environment Plan

Current recommendation: keep the benchmark simple unless a method truly forces
special dependencies.

## Recommended setup

Use one main environment for the runnable six-method MVSEC benchmark:

| Purpose | Env name | Suggested requirements |
|---|---|---|
| Main MVSEC benchmark | `env_mvsec_benchmark` | `configs/envs/common.txt` plus the method files you actually need |
| MatrixLSTM legacy fallback | `env_matrixlstm_legacy` | only if the official TensorFlow optical-flow path is needed later |

The per-method requirement files are still checked in because they are useful as
dependency references, but they do **not** imply that six or seven separate
environments are the best workflow.

## Why this is the default

- one environment is easier to debug
- the current local benchmark loop is lightweight and CPU-friendly
- most of the runnable code path is NumPy-based right now
- only MatrixLSTM has a realistic chance of requiring a special environment

Notes:

- MatrixLSTM is special. The official optical-flow repo is TensorFlow/Docker.
  The file here is for a future PyTorch adaptation into the unified benchmark.
- OmniEvent code is still incomplete upstream. Its env file is provisional.
- Once the project moves to a GPU server, a single PyTorch environment is still
  the preferred default for `EST`, `ERGO`, `Event Pre-training`, `GET`, and
  `EvRepSL`.
