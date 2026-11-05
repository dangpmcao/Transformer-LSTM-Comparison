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
The local CPU environment uses PyTorch 2.7.1+cpu. Each run records its PyTorch
and CUDA versions and checks them on resume. A new run directory is needed after
changing either version.

The run folder contains `run.json`, `metrics.jsonl`, `summary.json`, best and
final model weights, and `latest.pt` with optimizer, random-generator, epoch,
window-cursor, and token-budget state. Restart the same command with `--resume`
after interruption. The trainer refuses to overwrite a nonempty run folder
unless resuming, and rejects changed data, model code, or run settings.

## Production runs

After both Transformer and LSTM timing trials produce a valid pilot report, use
`comparison.experiment_settings --pilot-report ... --snapshot-file ...` to create
a production-ready snapshot. Then use `--mode production --snapshot-file ...`
with a production seed and the architecture's selected S tuning settings. The
trainer reads the exact token budget and microbatch size from that snapshot and
rejects a preparation-only snapshot. Test data is not used during training.
The proposal's 250M tokens is a target until the six-model timing pilot fixes
the exact budget; do not treat an exploratory run as a production result.
