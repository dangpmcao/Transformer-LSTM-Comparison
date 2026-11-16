# LSTM language model

`model.py` implements the classic LSTM side of the comparison: one configurable
model for sizes S/M/L plus the 4-layer depth control at M. Every size trains
independently from scratch with the **same training loop as the Transformer**
(`python -m transformer.train --architecture lstm`).

## Architecture and why

```
token IDs → Embedding(16384, d) → dropout → LSTM (2 layers, hidden h, dropout between layers)
          → linear adapter h→d → dropout → tied output head (Eᵀ) → logits [batch, time, 16384]
```

| Decision | Choice | Basis |
|---|---|---|
| Recurrent cell | Classic `nn.LSTM`, 2 layers | Proposal: "classic two-layer LSTM". Results apply to classic LSTMs, not xLSTM-style variants |
| Tied embeddings | Output head reuses the input embedding matrix | Proposal: "input/output embeddings are tied in both families"; Press & Wolf (2017), Inan et al. (2017) |
| Output adapter | Linear h→d (with bias), no activation, after the recurrence | Tying needs a d-dimensional output, but the proposal uses h ≠ d. The TA asked to include "any projection needed to tie embeddings when LSTM hidden and embedding dimensions differ"; `protocol.yaml` fixes `include_lstm_output_adapter: true`. Linear is the minimal map |
| Not `nn.LSTM(proj_size=…)` | Rejected | LSTMP projects the hidden state *inside* the recurrence; that is no longer a classic LSTM |
| Not AWD-LSTM style (last layer h = d) | Rejected alternative | Merity et al. (2018), the model analysed by Khandelwal et al. (2018), avoids the adapter by giving the last layer h = d, but that changes the proposal's architecture |
| State | Fresh zero hidden and cell state every window | Matches the Transformer's per-window history (`protocol.yaml`: reset each window) |
| Initialization | N(0, 0.02) matrices, zero biases, forget-gate input bias 1, PAD row 0; **token embedding U(−0.1, 0.1)** | `protocol.yaml` `training.initialization`, with the LSTM embedding exception `lstm_token_embedding` ([why](#finding-normal0-002-embeddings-stall-the-lstm-at-unigram-loss)). U(±0.1) is the usual embedding init of recurrent LMs (AWD-LSTM code of Merity et al. 2018; PyTorch `word_language_model`) |
| Dropout | Embeddings, between LSTM layers, adapter output; no recurrent dropout | `protocol.yaml` `training.dropout.lstm` |

## Sizes and exact parameter counts

Hidden widths were re-derived from the executable models so that non-embedding
parameters (LSTM + adapter; the tied head counted once) match the Transformer's
non-embedding parameters. Protocol tolerance is ±5%; all sizes are within 0.2%.

| Size | Layers | d | h (proposal) | h (used) | LSTM non-embedding | Transformer non-embedding | Difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| S | 2 | 256 | 460 | **472** | 3,285,376 | 3,290,624 | −0.16% |
| M | 2 | 384 | 864 | **873** | 10,836,060 | 10,844,160 | −0.07% |
| L | 2 | 640 | 1680 | **1690** | 39,708,880 | 39,717,120 | −0.02% |
| M deep | 4 | 384 | 580 | **588** | 10,828,992 | 10,844,160 | −0.14% |

Why the widths changed: the proposal's table did not include the adapter. With
it, the proposal's h = 460 at S lands at −4.71%, at the edge of the tolerance.
The token embedding (16,384 × d) is identical to the Transformer's at each size.
`tests/test_lstm.py` instantiates both models and checks these counts.

## Timing pilot (2026-11-14)

2M-token exploratory runs on one idle GPU of the shared server (24 GiB),
PyTorch 2.7.1+cu126, float32, effective batch 64 windows, lr 1e-3, dropout 0.1, seed 101.

| Size | tok/s, microbatch 2 | tok/s, microbatch 16 | Peak CUDA memory (mb 16) | Validation pass (mb 16) | `latest.pt` |
|---|---:|---:|---:|---:|---:|
| S | 21,793 | **161,745** | 2.56 GB | 0.54 s | 89.8 MB |
| M | 21,546 | **123,271** | 2.90 GB | 0.66 s | 205.6 MB |
| M deep | – | **107,454** | 3.01 GB | 0.88 s | 205.5 MB |
| L | 6,109 | **45,919** | 3.80 GB | 2.19 s | 602.4 MB |

**Finding: microbatch 2 starves the LSTM.** With only 2 windows per forward
pass, each of the 512 sequential LSTM steps is too small to occupy the GPU, so S
and M run at the same speed. Microbatch 16 (× 4 accumulation, same effective
batch of 64) is 7–7.5× faster and gives the same validation NLL after 2M tokens
(S: 7.411 at both settings), as expected: gradient accumulation leaves the
optimization unchanged apart from floating-point summation order.

Estimated GPU time at 250M tokens per run (microbatch 16, excluding validation):
S 26 min, M 34 min, M deep 39 min, L 1 h 31 min. The original 8-trial tuning grid at
50M tokens per trial: ~41 min. Full LSTM plan (tuning + 3 seeds × 4 sizes):
**~10–11 GPU-hours**, versus roughly 70 GPU-hours at microbatch 2.

## Finding: normal(0, 0.02) embeddings stall the LSTM at unigram loss

The first tuning grid (2026-11-14; size S, seed 101, 50M tokens per trial,
microbatch 16 × 4) used the shared initialization N(0, 0.02) for every matrix,
embedding included. Validation NLL at 50M tokens:

| lr \ dropout | 0 | 0.1 |
|---|---:|---:|
| 1e-4 | 7.343 | 7.343 |
| 3e-4 | 6.497 | 6.475 |
| 1e-3 | 6.097 | 6.105 |
| 3e-3 | 6.176 | 6.377 |

**These results are superseded.** Every trial sat at NLL ≈ 7.39 for about the
first 10M tokens: the loss of a unigram model, i.e. the LSTM was not using the
context. lr 1e-4 never left that plateau. The grid therefore ranked how fast each
learning rate *escaped the plateau*, not which one trains best.

Cause: the Transformer applies LayerNorm to its input, so the embedding scale does
not matter there. The LSTM feeds the embedding straight into its gates; with
entries of size ~0.02, times recurrent weights of size ~0.02, the context signal
is almost zero. Diagnosis (lr 1e-3, dropout 0, 10M tokens, one change at a time;
validation NLL):

| Initialization | 2M | 4M | 6M | 8M | 10M |
|---|---:|---:|---:|---:|---:|
| Protocol, N(0, 0.02) everywhere | 7.430 | 7.413 | 7.399 | 7.393 | 7.388 |
| Recurrent weights U(±1/√h) (PyTorch default) only | 7.428 | 7.414 | 7.399 | 7.393 | 7.388 |
| Recurrent U(±1/√h) + embedding U(±0.1) | 7.403 | 7.241 | 6.958 | 6.819 | 6.762 |
| **Embedding U(±0.1) only** (adopted) | 7.400 | 7.265 | 7.002 | 6.856 | 6.794 |

The embedding scale alone causes the stall. The fix changes only the LSTM token
embedding (and therefore its tied output head); all other LSTM parameters and the
whole Transformer keep the shared initialization. The first grid is kept on the
server as `lstm/runs/tuning_superseded_normal_init/`.

## Tuning (2026-11-16, fixed initialization)

Same grid and settings (S, seed 101, 50M tokens per trial, microbatch 16 × 4).
Validation NLL at 50M tokens, superseded → fixed initialization:

| lr \ dropout | 0 | 0.1 |
|---|---:|---:|
| 1e-4 | 7.343 → 7.159 | 7.343 → 7.108 |
| 3e-4 | 6.497 → 6.324 | 6.475 → 6.331 |
| 1e-3 | 6.097 → 5.717 | 6.105 → 5.656 |
| 3e-3 | 6.176 → 5.113 | 6.377 → **5.079** |

Every trial improves, and the ranking changes: the winner moves from lr 1e-3,
dropout 0 (6.097) to **lr 3e-3, dropout 0.1 (5.079)**, so the initialization bug
would have changed the tuning decision, not only the loss. The winner is the
largest learning rate in the protocol grid.

## Run

On the shared server, use the queue: at most 2 GPUs, only idle ones (≤5%
utilization and ≤2 GiB in use, sampled three times), jobs queued, `nice`/`ionice`,
finished runs skipped, interrupted runs resumed.

```bash
python -m lstm.queue pilot                                   # timing pilot S/M/L (2M tokens each)
python -m lstm.queue tuning --tuning-tokens 50000000         # 12-trial grid at S, microbatch 16 (agreed with Transformer side)
python -m lstm.queue production --learning-rate <lr> --dropout <p>   # seeds 13/37/73; S, M, M deep, L
python -m lstm.queue tuning --tuning-tokens 50000000 --dry-run       # show plan and idle GPUs only
```

Runs go to `lstm/runs/<pilot|tuning|production>/…` (ignored by Git), each with a
`<run>.log` next to the folder. Progress of every run:

```bash
python -m comparison.status --roots lstm/runs --watch 60
```

Production runs need `comparison/protocol.production.lock.json` (see the
comparison guide). Tuning uses exploratory mode, because production mode always
trains the full budget.
