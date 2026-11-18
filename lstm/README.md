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

### Grid extension (agreed with the Transformer side)

Because the winner sat at the grid edge, the shared grid was extended
symmetrically (protocol rule `grid_extension: symmetric_and_recorded_before_production`)
to six learning rates: 12 trials per architecture. Only the 4 new trials ran; the
other 8 settings are unchanged. Validation NLL at 50M tokens:

| lr \ dropout | 0 | 0.1 |
|---|---:|---:|
| 1e-4 | 7.159 | 7.108 |
| 3e-4 | 6.324 | 6.331 |
| 1e-3 | 5.717 | 5.656 |
| 3e-3 | 5.113 | 5.079 |
| 1e-2 | 4.543 | 4.556 |
| 3e-2 | 4.899 | **4.530** |

- The extension improves the best NLL by 0.55 nats (5.079 → 4.530).
- Dropout 0 is bracketed: loss rises from lr 1e-2 to 3e-2.
- **Selected (protocol rule, lowest validation NLL): lr 3e-2, dropout 0.1**, reused
  without retuning for M, M deep and L. Its margin over lr 1e-2 / dropout 0
  (4.543) is only 0.013 nats with one seed. lr 3e-2 is high for the larger sizes;
  any diverged production run is kept and reported (`failed_trials: retain_and_report`).

## Production (2026-11-16/17)

Snapshot `comparison/protocol.production.lock.json` (SHA-256 `47f139a9…`): 250M
exact prediction tokens, microbatch 16 × 4, lr 3e-2, dropout 0.1. Staged: L seed 13
first as a stability check, then the rest once it finished stable. Validation NLL
of the best checkpoint (lowest validation NLL, protocol selection rule):

| Size | seed 13 | seed 37 | seed 73 |
|---|---:|---:|---:|
| S | 4.008 | 4.005 | 4.008 |
| M | 3.758 | 3.765 | 3.776 |
| M deep | 3.890 | **diverged** | 4.045 |
| L | 3.786 | 3.761 | 3.586 |

### Finding: lr 3e-2 diverges for the 4-layer LSTM

M deep seed 37 stopped at 4.2M tokens with `FloatingPointError: Non-finite gradient
norm`, just after warm-up reached the peak lr of 3e-2; its loss was still at the
unigram plateau (~7.43). The trainer is deterministic, so rerunning the same seed
would diverge at the same point. Other signs that 3e-2 is at the edge of
stability: M deep seed 73 ends above its best checkpoint (final 4.069, best
4.045; training NLL 4.42), and the L seeds spread by 0.20 nats. The learning rate was
selected at S (2 layers) and, by protocol, reused without retuning.

Handling:
- **Primary result (as registered):** the run is kept and reported as diverged;
  M deep has 2 of 3 seeds. Picking another seed instead was rejected, because it
  would select seeds after seeing outcomes.
- **Declared sensitivity analysis (agreed with the Transformer side):** M deep,
  all three seeds, with the second-ranked tuning setting, lr 1e-2 and dropout 0
  (4.543 vs 4.530 at S), same snapshot and budget. Runs go to
  `lstm/runs/sensitivity_mdeep_lr0.01/` and never replace production runs.

Sensitivity result (2026-11-17; best-checkpoint validation NLL):

| M deep | seed 13 | seed 37 | seed 73 | Mean | Spread |
|---|---:|---:|---:|---:|---:|
| lr 3e-2, dropout 0.1 (production) | 3.890 | diverged | 4.045 | 3.968 (2 seeds) | 0.155 |
| lr 1e-2, dropout 0 (sensitivity) | 3.986 | 3.981 | 3.955 | 3.974 | 0.031 |

- At lr 1e-2 all three seeds train; the seed spread is 5 times smaller.
- The mean is almost the same (3.974 vs 3.968 over the two surviving seeds), so
  the divergence did not hide a better or worse M deep: the result does not depend
  on the learning-rate choice.
- The depth control holds under both learning rates: 4-layer M deep is about
  0.2 nats worse than the 2-layer M at the same non-embedding parameters
  (M at lr 3e-2: 3.758 / 3.765 / 3.776, mean 3.766). M was not run at lr 1e-2.

## Final evaluation (test, 2026-11-18)

`lstm/evaluate.py` scores every production and sensitivity run in two views:
**best_validation** (`best.pt`, the protocol's selection rule; primary) and
**final_budget** (`final.pt`, after exactly 250M tokens). Numerics: fp32, no
autocast, TF32 off, eval mode, fresh zero state for every input. M deep seed 37
(production) diverged and is listed as unscored in
[`results/campaign_summary.json`](results/campaign_summary.json).

Before any test scoring, each view must pass a **validation gate**: the evaluator
recomputes the validation NLL from the checkpoint and compares it with the training
record (tolerance 1e-4 nats). All 28 views pass; the largest difference is 3.6e-05.

Summaries (`summary.json`, `validation_check.json`) are versioned in
[`results/`](results/) as `<run>/<view>/{test,blimp,same_target,copy}/summary.json`.
They record the sha256 of the detail files (`windows.npz`, `pairs.jsonl`,
`per_target_nll.npz`), which stay outside git in `../results/lstm/` (server and a
backup copy next to the repo).

### Test perplexity (best_validation)

| Size | seed 13 | seed 37 | seed 73 | Mean | Context gain (mean) |
|---|---:|---:|---:|---:|---:|
| S | 56.68 | 56.31 | 56.75 | 56.58 | 0.400 |
| M | 44.16 | 44.42 | 45.12 | 44.57 | 0.441 |
| M deep | 50.70 | **diverged** | 59.09 | 54.89 (2 seeds) | 0.401 |
| L | 45.32 | 44.89 | 37.42 | 42.55 | 0.436 |
| M deep, lr 1e-2 (sensitivity) | 55.52 | 55.28 | 54.32 | 55.04 | 0.409 |
| 5-gram (reference) | | | | 66.45 | 0.183 |

final_budget means: S 56.83, M 44.82, M deep 55.64, L 42.75, M deep lr 1e-2 55.38.
The views differ by at most 1.4 perplexity points per run; L seed 13 and M deep
lr 1e-2 seed 37 have their best checkpoint at the final step (identical views).
The test ranking of the means matches validation: L < M < M deep < S. L seed 73 is the best
single run (37.4), and the L seed spread (7.9 points) is the largest; this agrees
with the instability at lr 3e-2 noted above.

### Same-target context ablation (test, best_validation)

The same 69,376 targets as the 5-gram (positions 385–512 of 542 full windows).
Values are the extra NLL (nats) of each reduced history over the full one; the
5-gram has exactly 0 in every column.

| Size | Full NLL | k = 4 | k = 32 | k = 128 | Distant shuffle (keep 128) |
|---|---:|---:|---:|---:|---:|
| S | 3.985 | +0.708 | +0.099 | +0.014 | +0.014 |
| M | 3.746 | +0.806 | +0.123 | +0.018 | +0.021 |
| M deep | 3.957 | +0.712 | +0.099 | +0.014 | +0.013 |
| L | 3.701 | +0.810 | +0.120 | +0.018 | +0.021 |
| M deep, lr 1e-2 | 3.960 | +0.702 | +0.104 | +0.014 | +0.011 |

- Almost all of the LSTM's context benefit comes from the last 32 tokens. Going
  from 128 tokens to the full history (up to 511) gives only 0.014–0.018 nats.
- Shuffling the history beyond the nearest 128 tokens costs about as much as
  removing it. So the small benefit of distant context does not depend on word
  order. This agrees with Khandelwal et al. (2018), *Sharp Nearby, Fuzzy Far
  Away*, for LSTM language models.
- The full-history values come from one forward pass per window. A check against
  explicit prefixes on 256 rows per view passes (largest deviation 1.0e-05).

### Copy(L) (test, best_validation)

Shared manifest (see `ngram/README.md`): `[BOS]+A+A` (copy) and `[BOS]+A+B`
(control). Copy gain = mean NLL of the first span minus mean NLL of the second.
Natural test text:

| Size | L = 25 | 50 | **100** | 200 | Copy − control at 100 | Copy gain at 100, positions 5+ |
|---|---:|---:|---:|---:|---:|---:|
| S | 0.173 | 0.117 | **0.068** | 0.029 | 0.385 | 0.101 |
| M | 0.234 | 0.157 | **0.096** | 0.044 | 0.452 | 0.132 |
| M deep | 0.132 | 0.098 | **0.061** | 0.026 | 0.391 | 0.097 |
| L | 0.216 | 0.147 | **0.090** | 0.041 | 0.453 | 0.128 |
| M deep, lr 1e-2 | 0.114 | 0.086 | **0.058** | 0.025 | 0.428 | 0.097 |
| 5-gram | −0.089 | −0.044 | **−0.024** | −0.012 | 0.005 | 0.000 |

Random-ID spans at L = 100: 0.17 (S), 0.25 (M), 0.19 (M deep), 0.18 (L), 0.10 (M deep lr 1e-2).

- The LSTMs copy only weakly: `natural_copy_100` is 0.06–0.10 nats, compared
  with a first-span NLL of about 4. The gain falls steadily with L (L: 0.22 at
  L = 25, 0.04 at L = 200), so the LSTM keeps little of a span that is 100 or
  more tokens away.
- Copy − control is much larger (0.39–0.45) because the control gain is
  negative: span B, which follows an unrelated span, is harder than span A, which
  follows BOS (about 0.3 nats, also beyond position 4). Copy − control therefore
  overstates copying, and the primary metric stays the copy gain itself.

### BLiMP (best_validation)

| Size | Overall | Without the 8 `?` paradigms | Equal-BPE-length pairs | Length-normalized | `matrix_question_npi_licensor_present` |
|---|---:|---:|---:|---:|---:|
| S | 62.2% | 64.8% | 62.9% | 60.9% | 0.2% |
| M | 63.0% | 65.4% | 63.8% | 61.5% | 0.2% |
| M deep | 60.7% | 63.1% | 61.7% | 59.9% | 0.2% |
| L | 62.1% | 64.4% | 62.9% | 60.9% | 0.3% |
| M deep, lr 1e-2 | 60.4% | 62.9% | 59.8% | 58.3% | 0.4% |
| 5-gram | 54.0% | 55.2% | 54.3% | 53.1% | 3.9% |

Seed means; within a size the seeds span 1.0–4.3 points (S 60.3–64.6%), larger than the
differences between sizes. Every LSTM beats the 5-gram by 6–9 points, but BLiMP
accuracy does not increase with size or perplexity: L is the best size on
perplexity and not on BLiMP. On the paradigm where only the acceptable sentence
has the unseen bare `?` (finding 3 in `ngram/README.md`), the LSTMs are almost
always wrong (0.2–0.4%): they give the untrained ID-34 row a very low
probability. The "without" column removes this artifact. Per-group and
per-paradigm accuracies are in each `blimp/summary.json`.

## Saved run records

`lstm/runs/` holds the records of every LSTM run (`run.json`, `metrics.jsonl`,
`summary.json` and the `<run>.log` next to each folder). Checkpoints (`*.pt`) are
not in Git; they stay on the training server.

| Folder | What it is |
|---|---|
| `production/{s,m,m_deep,l}/seed{13,37,73}` | Production runs (lr 3e-2, dropout 0.1); `m_deep/seed37` is the diverged run |
| `production/snapshot/` | `protocol.production.lock.json` (SHA-256 `47f139a9…`), `joint_pilot.json`, the Transformer pilot it was built from, and `protocol.yaml` exactly as frozen (SHA-256 `3ba167b6…`, commit `5337da6`) |
| `sensitivity_mdeep_lr0.01/` | Declared sensitivity analysis: M deep, lr 1e-2, dropout 0 |
| `tuning/` | 12-trial S grid with the fixed initialization (selection) |
| `tuning_superseded_normal_init/` | First 8-trial grid with N(0, 0.02) embeddings (superseded) |
| `pilot_mb16_5M/` | Timing pilot in the production snapshot (same method as the Transformer pilot) |
| `pilot/`, `pilot_mb16/` | Earlier timing pilots at microbatch 2 and 16 (2M tokens, old initialization) |

The `protocol.yaml` in `comparison/` changed after the snapshot (the Transformer
side's merge added `grid_extension_reason`); the LSTM runs were trained under the
archived copy above.

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

Final evaluation (after `comparison.same_target --split test` and
`comparison.copy_task`; one free GPU, runs in series, finished parts skipped,
so an interrupted campaign resumes where it stopped):

```bash
nice -n 10 python -m lstm.evaluate --device cuda:N            # every run, both views
nice -n 10 python -m lstm.evaluate --runs lstm/runs/production/s/seed13 --views best_validation
python -m lstm.evaluate --collect                             # rewrite campaign_summary.json only
python -m pytest lstm/tests -q
```

Production runs need `comparison/protocol.production.lock.json` (see the
comparison guide). Tuning uses exploratory mode, because production mode always
trains the full budget.
