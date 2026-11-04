# Transformer language model

`model.py` implements the Stage 4 decoder-only baseline: causal self-attention,
learned absolute positions, pre-layer normalization, GELU feed-forward blocks,
residual connections, and tied token/output embeddings. S/M/L use 4/6/8 blocks,
widths 256/384/640, and 4/6/10 heads. The feed-forward width is four times the
model width. Matrix weights start with normal(0, 0.02), normalization scales at
one, and biases at zero. Padding starts at zero but remains trainable through the
tied output head. The model accepts the shared `input_ids`, `attention_mask`, and
`position_ids` fields and returns logits `[batch, time, 16384]`.

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
