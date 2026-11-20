# LSTM and KenLM evidence for the proposal report

Stage 2 writing preparation, 2026-11-20. Read `report/proposal/proposal.md` and the report template. This memo is an evidence inventory, not a formal pipeline integrity certification. No training, inference, or research artifacts were changed.

## Selection and primary evidence

Use the nine `lstm_production_{s,m,l}_seed{13,37,73}` runs, `best_validation` view, against Transformer primary `mb16`. Each completed neural run consumed exactly 250,000,000 nonpadding prediction targets (EOS included; BOS and padding excluded). `final_budget` is a separately labelled sensitivity, not an extra replicate. LSTM `lstm/results/campaign_summary.json` contains both views and explicitly lists the diverged deep seed. Per-run source template below is `lstm/results/lstm_production_SIZE_seedSEED/best_validation/METRIC/summary.json`.

| Size | Test PPL, seeds 13/37/73 | Mean ± sample SD | BLiMP %, seeds 13/37/73 | Mean ± sample SD |
|---|---|---|---|---|
| S | 56.678694, 56.306715, 56.754586 | 56.5800 ± 0.2397 | 61.735821, 64.638806, 60.323881 | 62.2328 ± 2.2000 |
| M | 44.161576, 44.420781, 45.118793 | 44.5670 ± 0.4951 | 62.359701, 62.017910, 64.538806 | 62.9721 ± 1.3675 |
| L | 45.324240, 44.893762, 37.417234 | 42.5451 ± 4.4461 | 61.586567, 62.204478, 62.579104 | 62.1234 ± 0.5012 |

PPL source key: `test/summary.json:perplexity`; test NLL: `mean_nll`. BLiMP: `blimp/summary.json:primary.overall.accuracy` (multiply by 100). Seed means and sample SD above were recomputed from these records, not transcribed from rounded README values.

| Size | CG mean ± SD | Natural Copy(100) mean ± SD | Same-target k128 mean | Distant shuffle mean |
|---|---|---|---|---|
| S | 0.400204 ± 0.006262 | 0.067607 ± 0.002734 | 0.013598 | 0.014039 |
| M | 0.440930 ± 0.010685 | 0.096259 ± 0.000973 | 0.018444 | 0.020821 |
| L | 0.436354 ± 0.031043 | 0.090055 ± 0.033536 | 0.017929 | 0.020537 |

All gains are nats per target. CG source: `test/summary.json:context.context_gain`; copy: `copy/summary.json:natural_copy_100`; context interventions: `same_target/summary.json:truncation_gain.k128` and `shuffle_gain`. Early/late CG uses positions 1–32 minus 385–512 on 542 complete test windows. All 602 windows/291,710 targets enter test PPL. Same-target includes 69,376 targets, excludes 60 short windows/14,206 targets, and scores full/4/32/128/distant-shuffle contexts.

## Executable model and training controls

`lstm/model.py:MODEL_SIZES`, `LSTMLM`, and `parameter_counts` implement the classic LSTM, fresh zero hidden/cell states on every call, tied input/output embeddings and a linear adapter h→d outside the recurrence. This is not `nn.LSTM(proj_size=...)`. The adapter is included in nonembedding matching.

| Model | Layers | d | h | Nonembedding | Total | Relative nonembedding mismatch to Transformer |
|---|---:|---:|---:|---:|---:|---:|
| S | 2 | 256 | 472 | 3,285,376 | 7,479,680 | −0.15948% |
| M | 2 | 384 | 873 | 10,836,060 | 17,127,516 | −0.07469% |
| L | 2 | 640 | 1690 | 39,708,880 | 50,194,640 | −0.02075% |
| M deep | 4 | 384 | 588 | 10,828,992 | 17,120,448 | −0.13987% |

Counts are independently present in `lstm/runs/production/SIZE/seed13/summary.json:parameters` (M deep seed 13 is available). Transformer references are 3,290,624 / 10,844,160 / 39,717,120. Tests instantiate both executable definitions in `lstm/tests/test_lstm.py`, but tests were not run for this memo.

The proposal's h=460/864/1680 and deep h=580 were revised to 472/873/1690 and 588. Disclose actual dimensions, not the proposal estimates. Both architectures use `transformer/train.py`, fp32, AdamW, shared seeded per-epoch window order, effective batch 64, and primary microbatch 16 × accumulation 4. LSTM has a documented exception: token embedding initialization U(−0.1,0.1), while recurrent/adapter matrices remain N(0,.02). `lstm/model.py:_initialize` and `lstm/runs/production/snapshot/protocol.production.lock.json:configuration.training.initialization` are exact sources. Small normal embeddings stalled initial LSTM tuning; this architecture-specific change means the experiment is tightly controlled, but not literally an architecture-only change with identical initialization distributions.

The original eight-trial LR/dropout search was extended symmetrically to 12 trials per architecture: LRs 1e−4, 3e−4, 1e−3, 3e−3, 1e−2, 3e−2 × dropout 0/.1, each 50M targets, seed 101. LSTM selected .03/.1 (best S validation NLL 4.530), transferred without size-specific retuning. Transformer selected .001/0. Superseded normal-init trials are additional diagnostics, not the matched final tuning budget. Final identities: `lstm/runs/production/SIZE/seedSEED/run.json:identity`.

All 28 scored LSTM checkpoint views (11 completed production + 3 deep sensitivities, each best/final) passed the before-test validation recheck at tolerance 1e−4 nats. Evidence is every `validation_check.json:status,abs_difference`; do not equate these 28 views with 28 training seeds.

## Deep control and instability

Registered production M-deep seed 37 failed with `FloatingPointError: Non-finite gradient norm`; its log is `lstm/runs/production/m_deep/seed37.log`, with last logged validation at 4,019,988 consumed targets. There is no completed summary. Preserve the failure rather than replacing the seed. Registered deep results have 2/3 seeds: PPL 50.70 and 59.09, mean 54.89; these are not directly equivalent precision to three-seed means.

Declared sensitivity uses .01 LR, dropout 0, all original three seeds, same 250M budget, under `lstm/runs/sensitivity_mdeep_lr0.01` and matching `lstm/results/lstm_sensitivity_mdeep_lr0.01_*`. Mean test PPL ≈55.04. Four layers did not recover the two-layer M performance (44.57) at matched nonembedding count. This shows the observed Transformer benefit is not removed by this specific depth control. It does not prove recurrent depth never helps: hidden width changes to hold parameters fixed; primary deep is unstable/incomplete; the sensitivity changes two hyperparameters; two-layer M was not rerun at sensitivity LR.

## KenLM baseline

`ngram/train.py:train` runs unpruned interpolated modified Kneser–Ney estimation `lmplz -o 5` and an unquantized binary trie. Decimal BPE IDs are KenLM words, one training line per article, EOS mapped to `</s>`. Training source has 28,472 articles and 121,994,038 content tokens. `ngram/lm.py:KenLMScorer.target_log_probs` starts each evaluation window at null context, converts log10 probabilities using ln(10), counts/scorers OOV as `<unk>`, and never drops them. Null context avoids the article-heading artifact from `<s>` on mid-article windows.

`ngram/results/kenlm_5gram_train_test/summary.json`: PPL 66.4519489662, `mean_nll` 4.19647911454, targets 291710, windows 602, zero WikiText test OOV; `context.context_gain` 0.183314250575. The positive CG is compatible with only four-token history because initial positions have less available local context and token distributions vary with position. It is not evidence of long-history use.

`ngram/results/same_target_kenlm_5gram_train_test/summary.json` records exact per-target invariance under all history limits ≥4 and distant shuffle, maximum deviation zero. Full target mean NLL is 4.1708314544. `ngram/results/copy_kenlm_5gram_train_test/summary.json:natural_copy_100` is −0.0242229985053. Its `results.test_natural_text.100.exclude_first_four.copy_gain` is exactly 0; negative all-position gain comes from the four boundary positions and is not anti-copy capability.

`ngram/results/kenlm_5gram_train_blimp/summary.json:primary.overall` records approximately 54.0% BLiMP accuracy; 67,000 pairs and 72 ties including seven identical candidate pairs. Check exact primary count/accuracy from JSON for final table, not README rounded number. Equal-BPE-length sensitivity has 50,019 pairs.

## BLiMP measurement artifact and interpretable subgroup results

BLiMP is scored with summed subword log probabilities **including EOS**, strict good>bad comparison, ties incorrect. This is implemented in `comparison/blimp_scorer.py` and reused by LSTM/KenLM. Original text is preserved. `ngram/results/kenlm_5gram_train_blimp_coverage.json` records bare `?` ID34 in 15,000 candidate sentences, all absent in KenLM training; spaced question mark ID2885 is common in WikiText. Eight affected paradigms are listed in that record. KenLM maps missing ID34 to `<unk>` under declared treatment. Neural vocabulary coverage does not imply training support for ID34.

For `matrix_question_npi_licensor_present` (1,000 pairs), LSTM S correct counts by seed are 4,0,1; M 2,3,1; L 3,4,2. Transformer primary corresponding counts are S 0,0,1; M 6,15,2; L 1,6,2. Source: each BLiMP summary `primary.paradigms.matrix_question_npi_licensor_present`. These almost-zero accuracies in both families expose a format/coverage confound and should not be presented as pure grammar incapacity. If showing “Should Monica ever grin?” vs “Monica should ever grin.” from the documented analysis, label it an illustrative paradigm example; individual cross-model scores are not available here.

Descriptive group means (LSTM vs Transformer, percent; source each `primary.groups.GROUP.accuracy`): subject–verb agreement S 66.40 vs59.59, M68.63 vs61.66, L66.36 vs64.94; NPI licensing S43.83 vs43.35, M37.17 vs44.29, L32.63 vs42.24; determiner–noun agreement S76.24 vs73.77, M79.21 vs78.58, L77.98 vs82.62. These demonstrate heterogeneous grammar effects masked by overall scores; they are secondary descriptive breakdowns, without 12 additional confirmatory tests.

## Cross-architecture comparability and unavailable raw evidence

**Important copy deviation:** LSTM/KenLM use `comparison/copy_task.py:generator`, PCG64(SeedSequence([2718,source_index,length])) and one a/b draw per combination. Transformer uses `transformer/evaluation/manifests.py:build`, one PCG64(2718) streamed sequentially over sources and lengths. Both use uniform valid natural starts, 1,000 examples/length/source, no EOS/cross-article spans, but realized samples differ. Equal seed/count do not establish shared manifests. Report strong copy separation descriptively under the same sampling distribution; do not assert per-item pairing or use a shared-example cross-model bootstrap. Pairing *training seeds* is conceptually different and remains available, but the fixed between-architecture sample difference remains an explicit confound.

Same-target code is semantically identical: exhaustive full-window target positions 385–512 in article/window order; distant suffix 128 tokens preserved; permutation SeedSequence([1618,article_index,window_start,one_based_position]). LSTM histories exclude BOS in the iterator and its neural scorer adds BOS; Transformer includes BOS directly. Both reset recurrent state/positions for restricted histories. Full-history equivalence checks are saved for LSTM (256 rows, maximum ≈1e−5 nats).

LSTM source/result identities record token manifest hash `76509c...`, windows `5c52c...`; Transformer local manifests are `c90c17...` and `71e9ed...`. These differing metadata-file hashes do **not** prove different arrays. Both share tokenizer SHA256 `53af7083...`. `dataset/expected_outputs.lock.json` contains all 9 frozen NPY digests matching Transformer manifest inputs. However, archived LSTM locks include only metadata-manifest hashes, not per-array records; no LSTM-side historical byte verification record was found. The snapshot's joint pilot explicitly records both hosts' differing metadata hashes. Use shared pipeline/provenance and matching population counts as evidence, while avoiding a claim of independently reverified cross-host array identity.

The raw LSTM and ngram `pairs.jsonl`, `windows.npz`, `per_target_nll.npz` are absent in this checkout. The documented external root `../results/` does not exist, and no `pairs.jsonl` was found under baseline folders or the accessible parent tree. Versioned summaries retain their relative paths, byte sizes and hashes, but do not recover individual scores. Therefore verified cross-model same-ID qualitative disagreement examples and fresh paired item bootstrap cannot be supplied from these files. Do not fabricate either.

## Hypothesis conclusions and statistics boundaries

**Final-report inference supersedes the nominal arithmetic below.** `report/support/comparison_results.json` retains the original hypothesis-family sizes while assigning the untested copy slots p=1, because cross-architecture copy manifests differ. Under that conservative policy, confirmatory H1 CG Holm p-values are S .01299137, M .0007289703, L .01299137; H3 CG Holm p=.39603008. Copy effects remain descriptive. The numbers in the following bullets are the initial **nominal audit calculation only**, now preserved in `comparison_results.json:nominal_tests_for_audit_only`; their original Holm-adjusted p-values are **not** the final report's confirmatory inference. H2 TOST results and measured contrasts/intervals are unchanged. The original shared-sample copy conditions are not restored by the conservative multiplicity policy.

Nominal audit computed from saved best-validation summaries and paired training seeds using `transformer/evaluation/statistics.py:architecture_tests`, without scoring new data:

- T−L BLiMP mean differences S −2.65124 pp, M −1.47861 pp, L +1.84627 pp. Paired TOST at ±2 pp raw p=.705078/.318753/.428015; Holm across sizes gives .956260 each. None establishes equivalence. 90% CIs: S[−5.64119,.33870], M[−4.24645,1.28923], L[−.33550,4.02804]. Conclude H2 inconclusive, not equivalent, and do not infer a reliable grammar win from sign alone.
- CG T−L mean differences S .238778, M .250830, L .276960 nats, all positive; unadjusted seed-t 95% CIs [.164427,.313129], [.234003,.267656], [.182521,.371398]. Original six-endpoint H1 Holm family (including caveated copy) yields adjusted p .007795/.000607/.007795 for CG. Report family caveat rather than silently shrinking family to three favorable tests.
- CG scaling contrast (L−S architecture gap) .038181 nats, 95%CI[−.115111,.191474], raw one-sided p=.198015; natural-copy contrast .305823, CI[−.344748,.956394], raw p=.090231; two-endpoint Holm p .198015/.180462. H3 has a numerical trend but insufficient seed-level evidence.

The core interpretation is dissociation: large PPL/context/copy differences coexist with small, heterogeneous grammar differences that are too uncertain to certify ±2-point equivalence. Same-target evidence supports context sensitivity; it is not a causal decomposition showing that distant context explains most of the total perplexity gap. Truncation resets absolute positions and LSTM warmup; distant shuffle preserves input length/positions but also disrupts distant word forms at BPE boundaries.

## Compute and reproducibility constraints

LSTM production summaries contain `train_seconds`, throughput and `peak_cuda_bytes`; S seconds 2158.281/2227.103/2221.067, M1988.893/2012.981/1988.426, L6667.481/5370.275/5954.826; allocated bytes respectively 2,562,209,792 /2,900,477,440 /3,798,241,280. `lstm/runs/production/snapshot/joint_pilot.json:hardware` identifies LSTM RTX4090 D versus Transformer RTX3090 and explicitly says throughput is not cross-architecture-comparable. Keep time as consumed resources, not an architectural efficiency conclusion. No LSTM measured FLOPs counter was found; any FLOPs table must label an analytic estimate and its recurrent/output-head convention.

Main checkpoints are omitted from Git by `*.pt` and must be retrieved separately. Baseline artifact summaries are versioned but raw detail files and weights are unavailable locally. Reproduction commands exist in `lstm/evaluate.py`, `lstm/queue.py`, and ngram modules; avoid implying local baseline inference can run without obtaining those artifacts.
