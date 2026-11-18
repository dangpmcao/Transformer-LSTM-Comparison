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
nice -n 19 ionice -c3 python -m ngram.same_target --split test       # after comparison.same_target
nice -n 19 ionice -c3 python -m ngram.copy_task                      # after comparison.copy_task
python -m pytest ngram/tests -q
```

Defaults: KenLM binaries in `../.tools/kenlm/build/bin`, model in
`../models/ngram/kenlm_5gram_train/`, results in `../results/ngram/`. Outputs are
never overwritten. `summary.json` holds perplexity, OOV counts, the position
curve and context gain; `windows.npz` holds per-target NLL for each window
(for article bootstrap and model-disagreement analyses).

## Results

Summaries (`summary.json`) are versioned in [`results/`](results/); they record the
sha256 of the per-target/per-pair detail files, which stay outside git in
`../results/` (server and a backup copy next to the repo).

### Perplexity and position curve

| Metric | Validation | Test |
|---|---|---|
| Perplexity | 66.60 | **66.45** |
| Mean NLL (nats/token) | 4.199 | 4.196 |
| Targets / windows | 255,099 / 529 | 291,710 / 602 |
| OOV targets | 0 | 0 |
| Context gain (positions 1–32 minus 385–512) | 0.098 | **0.183** |
| Full windows (short excluded) | 469 (60) | 542 (60) |

Test position curve (full windows): 8.09, 6.06, 4.82, 4.28 at positions 1–4;
4.15 over 5–32, 4.19 over 33–384, 4.17 over 385–512. As on validation (finding 2),
the whole context gain comes from positions 1–4, where the history is still
shorter than four tokens; from position 5 on it is −0.03.

### Same-target context ablation (test)

69,376 targets from 542 full windows (60 short windows / 14,206 targets excluded).
Mean NLL 4.1708 in every condition (full, k = 4, 32, 128, distant shuffle);
per-target invariance **PASS**, maximum deviation exactly 0. The test manifest was
rebuilt because `protocol.yaml` changed after it was built (training/tuning
sections only); the rebuilt targets and shuffle permutations are byte-identical
to the superseded ones.

### Copy(L) (test)

Shared manifest from `comparison/copy_task.py` (protocol `evaluation.copy`):
1,000 examples per length and source, seed 2718. Each example scores
`[start, A, A]` (copy) and `[start, A, B]` (control, independent span B of the
same length). Gain = mean NLL of the first span minus mean NLL of the second.

| Source | L | Copy gain | Control gain | Copy gain, positions 5+ |
|---|---|---|---|---|
| Natural test text | 25 | −0.089 | −0.079 | 0.000 |
| Natural test text | 50 | −0.044 | −0.043 | 0.000 |
| Natural test text | **100** | **−0.024** | −0.029 | 0.000 |
| Natural test text | 200 | −0.012 | −0.011 | 0.000 |
| Random IDs | 25 | −0.055 | −0.057 | 0.000 |
| Random IDs | 50 | −0.028 | −0.020 | 0.000 |
| Random IDs | 100 | −0.014 | −0.004 | 0.000 |
| Random IDs | 200 | −0.007 | −0.004 | 0.000 |

`natural_copy_100` = −0.024: the 5-gram does not copy. From position 5 of the
repeated span it sees exactly the same four tokens as in the first span, so the
gain is exactly zero (an end-to-end check of the span alignment). The small
negative values come from positions 1–4: the first span starts from no history,
while the second starts after unrelated tokens, which costs back-off penalties.
Random-ID spans contain 309–2,402 tokens (per L) among the 99 IDs never seen in
train; they are scored as KenLM `<unk>`, as declared before scoring.

### BLiMP

**Overall accuracy 54.0%** (67,000 pairs; 72 ties, including 7 identical
candidates, counted incorrect). Sensitivities: equal-BPE-length pairs 54.3%
(50,019 pairs); length-normalized 53.1%.

| Group | Accuracy (%) | Pairs |
|---|---|---|
| anaphor_agreement | 46.4 | 2,000 |
| argument_structure | 63.3 | 9,000 |
| binding | 61.9 | 7,000 |
| control_raising | 64.0 | 5,000 |
| determiner_noun_agreement | 58.6 | 8,000 |
| ellipsis | 31.7 | 2,000 |
| filler_gap_dependency | 63.5 | 7,000 |
| irregular_forms | 60.2 | 2,000 |
| island_effects | 49.9 | 8,000 |
| npi_licensing | 30.5 | 7,000 |
| quantifiers | 41.3 | 4,000 |
| subject_verb_agreement | 54.6 | 6,000 |

Excluding the 8 paradigms affected by the OOV `?` (finding 3): 55.2% (59,000 pairs).
Per-paradigm accuracies are in `results/kenlm_5gram_train_blimp/summary.json`.

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
Test results are under Results above.

### Finding 3 — BLiMP's attached `?` never occurs in WikiText train

**What happened.** The coverage check required by the protocol
(`evaluation.kenlm.oov`) found exactly one missing ID before any BLiMP scoring:
bare `?` (ID 34), in 15,000 sentences. BLiMP attaches it to the word (`grin?`),
while WikiText always spaces punctuation (` ?`, ID 2885, 12,026 times in train).

**Affected paradigms.** Seven island/question paradigms, where both sentences of
a pair end in `?` (nearly symmetric), and `matrix_question_npi_licensor_present`,
where only the acceptable sentence ends in `?` (`Should Monica ever grin?` vs
`Monica should ever grin.`). The 5-gram scores 3.9% there.

**Treatment (declared before seeing any accuracy).** Keep the original BLiMP text
for every model; score the ID as KenLM `<unk>`, never drop it; report every model
with and without the 8 paradigms (computed from `pairs.jsonl`). `protocol.yaml`
was not changed, because its hash binds existing manifests and production locks.
The first failing check is kept as
`results/kenlm_5gram_train_blimp_coverage_UNDECLARED_first_check.json`.

**Lesson.** This is not a 5-gram problem: neural models never trained the
ID-34 embedding or output row either. Text normalization differences between
the training corpus and a benchmark show up as unseen token IDs.

## Pending

- Article/pair bootstrap intervals: computed in the shared comparison step for all models.
