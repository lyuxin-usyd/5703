# Environment Plan

The user asked to keep methods separated by virtual environment.

At this stage we are not creating the virtual environments automatically. We are
checking in per-method requirement files so each method can later be installed
in an isolated env without mixing dependencies.

Recommended env names:

| Method | Env name | Requirements file |
|---|---|---|
| EST | `env_est_mvsec` | `configs/envs/est.txt` |
| ERGO | `env_ergo_mvsec` | `configs/envs/ergo.txt` |
| Event Pre-training | `env_pretrain_mvsec` | `configs/envs/event_pretraining.txt` |
| GET | `env_get_mvsec` | `configs/envs/get.txt` |
| MatrixLSTM | `env_matrixlstm_mvsec` | `configs/envs/matrixlstm.txt` |
| EvRepSL | `env_evrepsl_mvsec` | `configs/envs/evrepsl.txt` |
| OmniEvent | `env_omnievent_mvsec` | `configs/envs/omnievent.txt` |

Notes:

- MatrixLSTM is special. The official optical-flow repo is TensorFlow/Docker.
  The file here is for a future PyTorch adaptation into the unified benchmark.
- OmniEvent code is still incomplete upstream. Its env file is provisional.
