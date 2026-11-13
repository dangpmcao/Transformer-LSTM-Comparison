# Transformer language model

`model.py` implements the Stage 4 decoder-only baseline: causal self-attention,
learned absolute positions, pre-layer normalization, GELU feed-forward blocks,
residual connections, and tied token/output embeddings. S/M/L use 4/6/8 blocks,
widths 256/384/640, and 4/6/10 heads. The feed-forward width is four times the
model width. Matrix weights start with normal(0, 0.02), normalization scales at
one, and biases at zero. Padding starts at zero but remains trainable through the
tied output head. The model accepts the shared `input_ids`, `attention_mask`, and
`position_ids` fields and returns logits `[batch, time, 16384]`.
Every window starts at position zero. The attention mask must describe a
nonempty, right-padded prefix; causal attention prevents future tokens from
affecting valid predictions. Padding targets are excluded by the shared loss
mask.

| Size | Blocks | Width | Heads | Token embedding | Other parameters | Total parameters |
|---|---:|---:|---:|---:|---:|---:|
| S | 4 | 256 | 4 | 4,194,304 | 3,290,624 | 7,484,928 |
| M | 6 | 384 | 6 | 6,291,456 | 10,844,160 | 17,135,616 |
| L | 8 | 640 | 10 | 10,485,760 | 39,717,120 | 50,202,880 |

These are exact counts for the 16,384-token vocabulary and 512 learned
positions. The proposal's 7.3M/16.9M/49.8M figures are approximate. The output
head shares the token-embedding weights, so it adds no separate parameters.
"Other parameters" includes learned positions and is the count used for the
protocol's 5% LSTM matching check.

## Completed training

**S/M/L training complete**, verified on 2026-11-13. All three exploratory runs
finished on 2026-11-11 with seed 101, exactly 250,000,000 prediction tokens and
8,087 optimizer updates per model.

| Size | Status | Final validation NLL | Final validation perplexity | Run summary |
|---|---|---:|---:|---|
| S | Complete | 4.110006 | 60.95 | [summary.json](runs/s/exploratory_250M_seed101/summary.json) |
| M | Complete | 3.773657 | 43.54 | [summary.json](runs/m/exploratory_250M_seed101/summary.json) |
| L | Complete | 3.453102 | 31.60 | [summary.json](runs/l/exploratory_250M_seed101/summary.json) |

Each local run folder contains `run.json`, `metrics.jsonl`, `summary.json`,
`best.pt`, `final.pt`, and `latest.pt`. Completion was checked against
`summary.json` and the final validation entry in `metrics.jsonl`; the table
reports final-budget validation metrics. Run artifacts are local and ignored
by Git, so a fresh clone needs a separate copy of these files.

These completed runs use `mode: exploratory`. The shared protocol snapshot
still has `production_ready: false` and `pilot: null`; the production run matrix
with seeds 13, 37, and 73 remains pending.

### Recorded training measurements

The following values come from each completed run's `summary.json`, checked on
2026-11-13. All runs record PyTorch `2.7.1+cu126`, CUDA runtime `12.6`, seed 101,
learning rate 0.0003, dropout 0.1, and microbatch size 2. Training uses float32
and 64 windows per effective batch (32 microbatches for a full optimizer step;
epoch-end and budget-end steps can be smaller).

| Size | Prediction tokens/s | Training seconds | Peak allocated CUDA bytes | Final validation seconds | Latest checkpoint bytes |
|---|---:|---:|---:|---:|---:|
| S | 56,818.05 | 4,400.01 | 529,134,592 | 1.265 | 89,898,315 |
| M | 37,332.10 | 6,696.65 | 779,211,776 | 1.839 | 205,735,691 |
| L | 16,153.18 | 15,476.82 | 1,495,474,688 | 4.959 | 602,571,979 |

- Throughput is consumed prediction tokens divided by accumulated training-step
  seconds. It excludes validation and checkpoint saving; it is not end-to-end
  throughput or a separate benchmark with a fixed warm-up period.
- Peak memory is PyTorch `max_memory_allocated`, not total device usage or
  reserved memory. The counter resets at process start/resume, so the recorded
  value cannot establish the peak of an earlier interrupted process.
- Validation time is the last full validation pass over 255,099 targets in 529
  windows. Checkpoint size is the full `latest.pt` resume state, including
  optimizer and RNG state, rather than only model weights.
- These measurements document the completed Transformer runs. They do not
  supply LSTM measurements or independently identify each run's physical GPU;
  `run.json` records PyTorch/CUDA versions but no GPU model or UUID.

### Data hashes for cross-machine verification

SHA-256 values computed directly from the local array files on 2026-11-13.
Paths below are relative to the external data root's `processed/` folder.

| Array | SHA-256 |
|---|---|
| `tokens/train.tokens.npy` | `9d66498cbeef4ba911cf7d7eb19c92d9c0bf0082586ddc78b514269820bb55fc` |
| `tokens/train.offsets.npy` | `6fbbfac0f7cd7d076c1a741ce6316c5d2af6db92287875ca0b93036b1f3a439f` |
| `tokens/validation.tokens.npy` | `88e3325d715ad66e1c6a0d89d0d03afaa672e63a7b44b639d8f4f216107cdd8c` |
| `tokens/validation.offsets.npy` | `49aa563e2f6b77def8a1bf4f466b45742ee88b60042b269da1bbf2226611258c` |
| `tokens/test.tokens.npy` | `0d56c6c2ba6c941e1769db8c65de464f2e4292e4000a182078548c3835ce79c4` |
| `tokens/test.offsets.npy` | `0e62df5bc83fe212302110ac7ca4e426ff9e2e5ecf8383ece514f92646fdc522` |
| `tokens/windows/train.windows.npy` | `c0e2b47aec16da2e07b5eeef53df96e6c0368afcc9043d70fd93a47a020387ce` |
| `tokens/windows/validation.windows.npy` | `40bb8f513930a5a15b8b02d36bebb6784930caea0a235f592aa923c6b49bd857` |
| `tokens/windows/test.windows.npy` | `876c3a43086bbf0ebfd74ea25049322566d384bc65e26ba45beecf69dbe553de` |

Recompute on this machine (CPU only):

```bash
cd /mnt/disk1/backup_user/dang.cpm/NLP/dataset/processed
sha256sum tokens/*.npy tokens/windows/*.npy
```

Cross-machine verification **confirmed on 2026-11-13**: all nine array-file
SHA-256 values and byte sizes match
[`dataset/expected_outputs.lock.json`](../dataset/expected_outputs.lock.json)
from main commit `fc1f5cc5f6818a3ef139182d15627ab582cc4a33`. The tokenizer
SHA-256 also matches. The local reference file now has `status: confirmed`;
its reference hashes are unchanged. Manifest-file hashes also include metadata
such as timestamps; compare the array-file hashes for this check.

## Run an exploratory timing/training trial on one GPU

From the repository root on a Linux server with Python 3.12, an NVIDIA GPU,
the prepared `../dataset/processed/tokens/` folder, and the trained tokenizer:

```bash
bash transformer/setup_gpu.sh
source .venv/bin/activate
python -m pytest transformer/tests -q
python -m transformer.train --mode exploratory --size s \
  --token-dir ../dataset/processed/tokens \
  --budget-tokens 1000000 --microbatch-windows 2 \
  --learning-rate 0.0003 --dropout 0.1 --seed 101 \
  --run-dir transformer/runs/s/exploratory_seed101
```

This uses float32, an effective batch of 64 windows, exact target-token
accounting, the shared validation split, and a token-based cosine schedule.
Microbatch 2 is a conservative starting point for a 24 GB GPU; actual memory and
speed are reported in `summary.json`. It is a single Transformer measurement, not
the six-measurement pilot required to unlock production in the shared protocol.
The project `requirements.txt` pins a CPU PyTorch wheel, so run `setup_gpu.sh`
instead of the data-preparation environment script on the server. The setup
script defaults to PyTorch 2.7.1 CUDA 12.6; set `PYTORCH_CUDA_INDEX` to the
appropriate official wheel index for the server driver if needed.
The script downloads the selected PyTorch wheel from that index and resolves
its dependencies from PyPI, avoiding NVIDIA download links that can time out.
Completed downloads are cached under `.venv/pip-cache`; failed install/download
operations are retried up to three times. Socket timeout and connection retries
default to 600 seconds and 10; override them with `TRANSFORMER_PIP_TIMEOUT` and
`TRANSFORMER_PIP_RETRIES`.
The local CPU environment uses PyTorch 2.7.1+cpu. Each run records its PyTorch
and CUDA versions and checks them on resume. A new run directory is needed after
changing either version.

The run folder contains `run.json`, `metrics.jsonl`, `summary.json`, best and
final model weights, and `latest.pt` with optimizer, random-generator, epoch,
window-cursor, and token-budget state. Restart the same command with `--resume`
after interruption. The trainer refuses to overwrite a nonempty run folder
unless resuming, and rejects changed data, model code, or run settings.

## Submit the 250M-token exploratory runs with Slurm

On this shared server, let Slurm allocate GPUs. From the repository root:

```bash
mkdir -p transformer/runs/slurm
env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL sbatch transformer/train_slurm.sbatch
```

The array maps tasks 0/1/2 to S/L/M and allows at most two tasks at once, each
requesting one RTX 3090, four CPU cores, 12 GB host memory, and 24 hours on
`vishc-server-1`, where the existing checkout and data live. Physical GPU IDs
come from Slurm. These remain exploratory runs with seed 101 and 250M total
prediction tokens per size. Existing `latest.pt` files resume the same run;
otherwise the trainer starts a new run and refuses a nonempty run folder.
The launcher locks each run to prevent duplicate Slurm writers and saves a
resume checkpoint every 25 updates. Model, data, seed, optimizer, learning rate,
dropout, microbatch, and total token budget match the original manual launch.
Resuming appends to the existing metrics, so entries after the last saved
checkpoint can appear again when that work is replayed.

Monitor with `squeue -u "$USER"` and the per-task `.out`/`.err` files under
`transformer/runs/slurm/`. Do not also run the old manual launch loop.
The batch jobs continue after SSH, VS Code, or tmux closes. They opt into Slurm
requeue after a node failure and resume from `latest.pt` when restarted. This
does not automatically retry training errors or extend the 24-hour time limit.

## Production runs

After both Transformer and LSTM timing trials produce a valid pilot report, use
`comparison.experiment_settings --pilot-report ... --snapshot-file ...` to create
a production-ready snapshot. Then use `--mode production --snapshot-file ...`
with a production seed and the architecture's selected S tuning settings. The
trainer reads the exact token budget and microbatch size from that snapshot and
rejects a preparation-only snapshot. Test data is not used during training.
The proposal's 250M tokens is a target until the six-model timing pilot fixes
the exact budget; do not treat an exploratory run as a production result.
