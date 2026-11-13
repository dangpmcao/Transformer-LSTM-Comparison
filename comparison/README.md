# Experiment settings and shared training rules

This folder defines how the future Transformer and LSTM experiments will be
compared. The rules are written in [`protocol.yaml`](protocol.yaml).
*Protocol* means the agreed experiment settings; a *snapshot* is a saved copy
of those settings together with the exact input/code checksums.

The data and shared input code are ready. Transformer S/M/L exploratory training
is complete at 250M prediction tokens per model, seed 101; see the
[completed training results](../transformer/README.md#completed-training).
The joint Transformer/LSTM timing pilot and production run matrix remain pending.
Run commands from the repository root, using the
[Python environment](../README.md#set-up-python).

## Save or verify a settings snapshot

First follow the [dataset guide](../dataset/README.md) to prepare token arrays,
prediction-window indexes, and the model-input check. Then save the settings:

```bash
python -m comparison.experiment_settings
```

This creates `comparison/protocol.lock.json` after verifying the tokenizer,
token arrays, and window indexes. It records their identities, the settings,
and the shared implementation hashes. It refuses to replace an existing file.
To verify a snapshot that already exists:

```bash
python -m comparison.experiment_settings --check
```

The snapshot is generated and ignored by Git. Archive it with the future runs
that use it. The [naming guide](../NAMING.md) maps the readable public commands
and names to their earlier equivalents; both interfaces use the same code.

## Rules shared by both models

| Area | Current decision |
|---|---|
| Tokenizer | Same trained 16,384-ID byte-level BPE |
| Prediction examples | Independent windows with up to 512 targets; never cross an article boundary |
| Sequence boundaries | BOS at each window; one EOS per article; padding does not count toward loss |
| Model memory | Restart positions and both LSTM states at each window |
| Training order | Same run seed, epoch, and batch size for both architectures |
| Run seeds | 13, 37, 73; separate tuning seed 101 |
| Effective batch | 64 windows per optimizer update; identical partial-batch handling |
| Numerical precision | float32 |
| Optimizer | AdamW; betas (0.9, 0.95), epsilon 1e-8, weight decay 0.1 except biases/norm parameters |
| Learning-rate schedule | Cosine decay based on consumed prediction tokens; 1% warmup; minimum LR is 10% of peak |
| Gradient clipping | Global norm 1.0 after loss normalization |
| Weight initialization | Matrix weights normal(0, 0.02), normalization scales 1, biases 0; LSTM forget bias totals 1 |
| Hyperparameter search | Four learning rates × two dropout values: eight trials per architecture at size S |
| Larger sizes | Reuse each architecture's selected S settings without extra tuning |
| Checkpoint selection | Lowest validation mean prediction loss; earlier checkpoint wins ties |
| Validation timing | Every 1M prediction targets, at the first optimizer step reaching each threshold, plus start/end |
| Parameter matching | Tied token embeddings/output head; same embedding width; other parameter counts within 5% of Transformer |

These are prospective experiment choices, not measured optimal settings.
Dropout locations and random-seed/resume requirements are also recorded in the
YAML. Stage 4 must implement the models and verify causal attention, state
resets, numerical behavior, and parameter counts.

## Load the same examples for either model

```python
import torch
from dataset.common import DEFAULT_DATA_ROOT
from dataset.model_inputs import (
    NextTokenWindowDataset, create_batch_loader, create_lstm_zero_state,
)

training_windows = NextTokenWindowDataset(
    DEFAULT_DATA_ROOT / "processed/tokens", "train",
)
training_batches = create_batch_loader(
    training_windows, batch_size=2, training=True, seed=13, epoch=0,
)
batch = next(iter(training_batches))
assert batch["input_ids"].shape == (2, 512)

hidden_state, cell_state = create_lstm_zero_state(
    batch["input_ids"], layers=2, hidden_width=460, dtype=torch.float32,
)
```

See the dataset guide for every batch field and a concrete shifted-input example.
Future logits have shape `[batch_size, 512, 16384]`. Move tensors together to the
model's device. The LSTM zero-state helper creates fresh hidden **and** cell
states; call it for each batch. Transformer positions start at zero each time.

Training shuffles examples using NumPy PCG64 with `SeedSequence([seed, epoch])`.
This uses a separate generator and does not consume the model's PyTorch random
state. Validation/test order is fixed. No final batch is dropped. A future
checkpoint must save the epoch and consumed-window position to resume that order.

## Compute loss and count training tokens

Import these functions from `dataset.model_inputs`:

| Function | What it returns |
|---|---|
| `compute_masked_cross_entropy(logits, targets, loss_mask)` | Summed natural-log prediction loss and the number of enabled targets |
| `limit_loss_mask_to_budget(loss_mask, remaining_tokens)` | A mask keeping only the remaining allowed targets, in batch-major order |

Mean loss is **total loss / total valid target count**. For validation/test,
add both quantities across all batches before dividing; perplexity is the
exponential of that mean. Do not average batch perplexities.

When accumulating gradients across microbatches, normalize by the entire
optimizer update's valid-target count before clipping. Different microbatches
can have different valid counts because of padding. Partial batches at epoch
boundaries or the final budget use their actual valid count for both models.
EOS counts toward the budget; BOS and padding do not. The budget-mask helper
allows the final update to stop at the exact number of targets. Do not run a
loss calculation or optimizer update when no targets remain.

## What the final evaluation will measure

| Evaluation | Saved rule |
|---|---|
| Text prediction | Mean prediction loss and perplexity across all valid test targets |
| Context benefit | Early-position loss minus late-position loss on all full test windows |
| Copy benefit | First-span loss minus repeated-span loss; primary span length is 100 tokens |
| Copy sampling | Fixed seed 2718; 1,000 examples per length/source; lengths 25, 50, 100, 200 |
| Grammar | BLiMP summed subword log probability including EOS; ties count as incorrect |
| Uncertainty | Training seeds are independent units; evaluation-example uncertainty is reported separately |
| Hypothesis tests | Prespecified directions, Holm correction groups, and ±2-percentage-point grammar equivalence tests |

The YAML specifies the exact position ranges, history restrictions, random-token
sampling, secondary analyses, and bootstrap settings. Short windows contribute
to overall perplexity but are excluded from the full-window position curve.
No test performance may select hyperparameters or checkpoints.

## Score a trained model on BLiMP

[`blimp_scorer.py`](blimp_scorer.py) implements the grammar evaluation. It needs
the BLiMP files from `python -m dataset.download` and `python -m dataset.prepare`,
but not the WikiText token arrays. A pair is correct only when the grammatical
sentence receives a strictly higher summed log probability (Warstadt et al., 2020).
Run it only after validation has selected the checkpoint.

```python
from comparison.blimp_scorer import evaluate_blimp, neural_window_scorer

model.eval()
score_windows = neural_window_scorer(
    lambda batch: model(batch["input_ids"]), device="cuda", batch_size=64,
)
summary = evaluate_blimp(
    score_windows, "results/blimp/transformer_s_seed13", model_label="transformer_s_seed13",
)
print(summary["primary"]["overall"]["accuracy"])
```

The adapter passes the same batch fields as the dataset loader, padded to the
longest window in the batch instead of always 512. The function must return logits
`[batch, time, vocab]`; for an LSTM, it must also create fresh zero states on each
call. Other models, such as KenLM, can replace the adapter with a function that
receives target-ID windows and returns one summed natural-log probability per
window. Each window implicitly starts from BOS, which is never scored.

The output folder is new and is never overwritten:

| File | Contents |
|---|---|
| `summary.json` | Accuracy overall and for 12 groups and 67 paradigms, both sensitivity analyses, tie and overlength counts, and input hashes |
| `pairs.jsonl` | One record per pair: both scores, both target counts, and each decision; use these for bootstrap and model-disagreement analyses |

## Why production training is not ready yet

The existing snapshot has **`production_ready: false`**. The plan requires a
short measured *timing trial* (also called a pilot) before fixing the exact
training budget, tuning-trial budget, microbatch size, and gradient-accumulation
count. **250M prediction targets is currently a target, not a finalized budget.**

This readiness check intentionally fails until those measurements exist:

```bash
python -m comparison.experiment_settings --check --require-ready
```

After implementing the models and measuring the timing trial, save a new snapshot:

```bash
python -m comparison.experiment_settings --pilot-report /path/to/pilot.json \
  --snapshot-file comparison/protocol.production.lock.json
```

The pilot report must contain:

- `status: "PASS"` and a nonempty `hardware` description.
- Matching `protocol_sha256`, `token_manifest_sha256`, and `window_manifest_sha256`.
- Six `measurements` records: Transformer/LSTM at S/M/L, each with `architecture`,
  `size`, and positive `prediction_tokens_per_second`, `peak_memory_bytes`,
  `validation_seconds`, and `checkpoint_bytes`.
- A `settings` object containing positive integers for `exact_prediction_tokens`,
  `tuning_prediction_tokens`, `microbatch_windows`, and
  `gradient_accumulation_steps`; microbatch × accumulation must equal 64.

Those saved field names remain unchanged for compatibility. The measured report
is copied into the new snapshot. Keep the earlier snapshot as preparation
history. Changes to shared rules require a new snapshot and relevant checks or
timing measurements before training.

## Custom locations and tests

The settings command accepts `--token-dir`, `--window-dir`, `--tokenizer-dir`,
`--settings-file`, and `--snapshot-file`. Each names the folder or file it reads
or writes. A moved cache remains usable when its contents and saved identities
are unchanged; absolute machine paths do not define the experiment identity.

```bash
python -m pytest dataset/tests tokenizer/tests comparison/tests -q
```

The dataset guide also explains array/index verification and the real-article
input/loss check. Tests use synthetic corpora and invented pilot records in
separate temporary directories; they do not establish measured training speed.
