# 5-gram baseline (KenLM)

A KenLM 5-gram trained on the **same WikiText-103 train token IDs** as the neural
models. It can use at most four previous tokens, so it is the "no long context"
control: it should show no late-context or copy gain beyond four tokens.

## Design

| Choice | Value | Why |
|---|---|---|
| Training data | WikiText-103 train only (28,472 articles, 121,994,038 content tokens) | Matched with the neural models. Warstadt et al. (BLiMP, 2020) trained their 5-gram on Gigaword; we cite them only as motivation for an n-gram baseline |
| Symbols | Shared 16,384-ID byte-level BPE; each ID is a KenLM word (`"523"`) | Per-token losses are directly comparable across models |
| Smoothing | Interpolated modified Kneser–Ney (`lmplz -o 5`), no pruning | KenLM's standard estimator; each order is interpolated with the lower orders |
| Training unit | One line per article; KenLM `</s>` = article EOS | Avoids false EOS events in the middle of articles |
| Evaluation start | **Null context** (no history) at every 512-target window | Mirrors the neural models' fresh state per window; see finding 1 |
| Log base | KenLM log10 × ln 10 → natural log | `comparison/protocol.yaml`, `evaluation.kenlm` |

KenLM commit `4cb443e60b7bf2c0ddf3c745378f76cb59e254e5`; binary trie without quantization (2.1 GB).

## Run

On a shared server, keep priority low:

```bash
nice -n 19 ionice -c3 python -m ngram.train --memory 16G          # ~7 min
nice -n 19 ionice -c3 python -m ngram.evaluate --split validation # sanity check
nice -n 19 ionice -c3 python -m ngram.evaluate --split test --blimp   # final reporting only
python -m pytest ngram/tests -q
```

Defaults: KenLM binaries in `../.tools/kenlm/build/bin`, model in
`../models/ngram/kenlm_5gram_train/`, results in `../results/ngram/`. Outputs are
never overwritten. `summary.json` holds perplexity, OOV counts, the position
curve and context gain; `windows.npz` holds per-target NLL for each window
(for article bootstrap and model-disagreement analyses).

## Results so far (validation)

| Metric | Value |
|---|---|
| Perplexity | 66.60 (mean NLL 4.199 nats/token, 255,099 targets) |
| OOV targets | 0 (16,284 of 16,380 non-control IDs occur in train) |
| Context gain (positions 1–32 minus 385–512, 469 full windows) | 0.098 |

Test perplexity and BLiMP are kept for final reporting.

## Findings log (for the report and presentation)

### Finding 1 — Scoring windows from `<s>` was a hidden artifact (fixed)

**What happened.** The first evaluation started every window from KenLM's `<s>`.
Because training used one line per article, `<s>` only ever preceded an article
start, and WikiText articles always begin with a ` = Title =` heading. The model
therefore learned "after `<s>` comes `=`". But most evaluation windows start in
the middle of an article.

**Evidence (validation, position 1):**

| Window start | NLL from `<s>` | NLL from null context |
|---|---|---|
| Article start (60 windows) | 0.00 | 6.65 |
| Mid-article (469 windows) | **17.99** | 8.27 |

| Metric | Before (`<s>`) | After (null context) |
|---|---|---|
| Position-1 mean NLL | 15.66 | 8.02 |
| Perplexity | 67.65 | 66.60 |
| Context gain | **0.325** | 0.098 |

About three-quarters of the original context gain came from this single
position. A metric meant to show "no long-context use" would have suggested the
opposite. The superseded result is kept on the server as
`results/ngram/kenlm_5gram_train_validation_SUPERSEDED_sentence_start/`.

**Fix.** Every window now starts from KenLM's null context: position 1 gets the
unigram probability, i.e. truly no history. Covered by
`test_windows_start_from_null_context_not_article_start`.

**Lesson.** "Start-of-sequence" must mean the same thing in training and
evaluation. The neural models see BOS at the start of *every* window, so their
BOS context is a mixture of article and mid-article starts; a model trained on
whole articles does not. Check the first positions of any position-wise curve.

### Finding 2 — Late positions are harder even for a 4-token-history model

After the fix, the 5-gram's mean NLL by position (validation, full windows):

| Positions | 1 | 2 | 3 | 4 | 5–32 | 33–384 | 385–512 |
|---|---|---|---|---|---|---|---|
| Mean NLL | 8.02 | 6.06 | 4.72 | 4.29 | 4.11 | 4.18 | **4.22** |

The 5-gram cannot use anything beyond four tokens, yet positions 385–512 are
*worse* than 5–32. So tokens at different positions differ in difficulty: later
text in a window (deeper into articles) is harder. Its whole context gain of
0.098 comes from positions 1–4, where history is still shorter than four
tokens; without them it would be about −0.11.

**Implication.** Early-versus-late loss mixes context use with position
difficulty. This is exactly the TA's concern, and it supports the planned
**same-target context ablation**: score the *same* target tokens with full versus
truncated or shuffled distant context, and report the 5-gram curve next to
the neural ones as the reference for "no long-context use".

## Same-target context ablation

`comparison/same_target.py` builds the shared target manifest defined in
`evaluation.context.same_target` (positions 385–512 of every full window;
histories full, 4, 32, 128 tokens, and a distant shuffle that keeps the nearest
128 tokens). `ngram/same_target.py` scores the 5-gram on that same manifest and
checks invariance **per target** (tolerance 1e-6 nats), as the protocol's
`five_gram_extension` requires.

```bash
python -m comparison.same_target --split validation   # persist manifest before any scoring
python -m ngram.same_target --split validation
```

Validation (60,032 targets from 469 full windows; 60 short windows / 14,971 targets excluded):

| Condition | full | k = 4 | k = 32 | k = 128 | distant shuffle |
|---|---|---|---|---|---|
| Mean NLL | 4.2227 | 4.2227 | 4.2227 | 4.2227 | 4.2227 |

Invariance **PASS**: maximum per-target deviation exactly 0. The full-history
NLL also equals the window evaluation's late-position mean (4.223), an
independent check that histories are aligned. This is the reference row for the
neural models: any gain they show here comes from context beyond four tokens.
The test manifest is built (69,376 targets) but not scored until final reporting.

## Pending

- Test perplexity, BLiMP and the test-split ablation (final reporting).
- Copy(L): needs the shared span sampler from `evaluation.copy` in the protocol (not yet implemented), so that all models score identical spans.
