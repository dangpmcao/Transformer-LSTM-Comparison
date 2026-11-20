# Transformer, tokenizer and shared-data evidence for the report

Prepared 2026-11-20 from current artifacts and source. This is Stage 2 writing support, not a completed pipeline integrity gate. No training or evaluation was rerun and no source artifacts were modified.

## Proposal and reporting scope

Read `report/proposal/proposal.md` including all TA feedback and `report/report_template.tex`. The proposal concerns architecture contrasts (H1 CG and natural Copy100; H2 grammar equivalence at +/-2 percentage points; H3 L-minus-S architecture-gap increase), not isolated Transformer scaling. Keep 18 primary neural runs (T/LSTM x S/M/L x three seeds), the M-deep recurrent control and KenLM central. The template requires 6--8 main-text pages, <=250-word abstract, its nine ordered numbered sections, contributions and reproducibility statements. Do not promote the historical seed-101 or extra batch campaign to extra independent seeds.

`transformer/evaluation/config.yaml` and saved `.../proposal_21/configuration.json` set primary campaign `mb16` (16 windows x 4 accumulation), sensitivity `mb64` (64x1), historical `initial` (seed101). Main seeds 13/37/73; `best_validation` is primary, `final_budget` a separately labelled sensitivity. There are 21 Transformer training runs and 42 checkpoint views; a checkpoint view is not a training replicate. `.../proposal_21/report.json` has 42 expected/completed, missing=[]; its architecture-comparison status is stale for today's combined checkout: it says external LSTM/KenLM needed because those were not imported into that collector. Do not mistake this for absent LSTM results today.

## Shared inputs and tokenizer

- Single 16,384-ID byte-level BPE with 4 special IDs, min_frequency=2, no normalization and no prefix-space insertion; train articles only. Source: `tokenizer/artifacts/manifest.json` -> config/data_identity; frozen SHA256 `53af7083db6636696cbd023afc4e7e07f7c2c4daaff3fe007e1bcf33fd5f6f70`.
- Train/validation/test article counts: 28,472/60/60. Content BPE tokens: 121,994,038/255,039/291,650. One EOS/article gives prediction targets 122,022,510/255,099/291,710. Sources: `tokenizer/artifacts/checks.json` -> splits; `../dataset/processed/tokens/manifest.json` -> splits.
- Frozen windows are nonoverlapping, up to 512 targets, no article crossings. Counts train/val/test 252,574/529/602. Full-window counts 224,144/469/542. Sources: `../dataset/processed/tokens/windows/manifest.json` -> splits. Each window gets BOS and position restart; LSTM state resets. Loss sums over valid targets then divides by valid target count; EOS counts, PAD/BOS do not.
- Tokenizer checks record PASS, exact roundtrip of every article, zero reserved IDs in content, 13 probes, BLiMP not accessed. Source: `tokenizer/artifacts/checks.json`; use `tokenizer.core.load_tokenizer` rather than raw HF reload because `encode_special_tokens=True` must be restored.
- Corpus source revision: Salesforce/wikitext `b08601e04326c79dfdd32d625aee71d232d685c3`; BLiMP `3e56b06fcabca9b30822fc66435fca6b1aa40bb1`. Source lock: `dataset/sources.lock.json` (also tokenizer manifest data_identity for WikiText).
- One official-split exact article duplicate retained: train:003821/test:000059, The Hustler (film). Source `../dataset/processed/duplicates.json` -> cross_split_duplicate_groups. Do not claim perfectly uncontaminated WikiText test set. No dedup was applied to preserve official splits.
- WikiText licensing is inconsistent upstream: pinned metadata says CC-BY-SA-3.0/GFDL, card prose says CC-BY-SA-4.0. `../dataset/ATTRIBUTION.json` explicitly preserves this disagreement; do not assert a single unqualified license. BLiMP CC-BY-4.0 confirmed in pinned README.

## Implemented Transformer and training

`transformer/model.py`: causal pre-LN decoder, learned absolute positions, GELU FFN width=4d, tied token embeddings/head, no output bias. Sizes (layers,width,heads): S=(4,256,4), M=(6,384,6), L=(8,640,10). Actual (token embedding/nonembedding/total) parameters:

| S | 4,194,304 | 3,290,624 | 7,484,928 |
| M | 6,291,456 | 10,844,160 | 17,135,616 |
| L | 10,485,760 | 39,717,120 | 50,202,880 |

Learned positional embeddings count in nonembedding matching; tied head counted once. Verified by each primary training summary and evaluation task `resources.parameters`. LSTM actual hidden widths differ from proposal sketches (472/873/1690 and M-deep588, not 460/864/1680/580) because output adapters enter matching counts. Current `lstm/model.py` comments say <0.2% mismatch; independently calculate actual counts before reporting exact percent.

Primary T run identity: 250,000,000 prediction targets, batch64, microbatch16, LR=.001, dropout0, float32, PyTorch2.7.1+cu126, CUDA12.6; 8,087 updates. Source `transformer/runs/transformer_only_250M_3seed_mb16_acc4/{s,m,l}/seed{13,37,73}/run.json` -> identity and summary.json. Training protocols use AdamW (.9,.95), eps1e-8, decay.1 excluding bias/norm, cosine token-based schedule with 1% warmup and minimum .1peak; norm clip1. Source comparison/protocol.yaml, unchanged trainer. All models reset each window. Trainer uses deterministic algorithms and saves RNG/order/optimizer in resumable checkpoints.

Tuning: 12 S trials, 6 LR choices .0001/.0003/.001/.003/.01/.03 x dropout0/.1; seed101, 50M targets each =600M targets per architecture if LSTM equal as expected. Selected Transformer trial04 (.001/0): best NLL4.5464441052/PPL94.30. Selection budget per trial50M; transfer to M/L without additional tuning. Source `transformer/runs/s/tuning_50M_seed101_6lr/study.json`, trial summaries and campaign tuning_selection.json. Proposal originally8 configurations; current protocol documents symmetric extension to12 (2026-11-16). Distinguish original plan and implemented procedure.

IMPORTANT matching caveat: current protocol and `lstm/model.py` explicitly use LSTM token-embedding U(-.1,.1), versus T N(0,.02), because initial small-normal LSTM embedding trials stalled at unigram loss. Report the exception; this is a controlled protocol comparison with architecture-specific initialization/tuning, not literally a pure single-factor architecture-only intervention. Both shared `comparison/protocol.yaml` and T evaluation manifest snapshot will carry differing protocol histories; check recorded identities, not hash equality inferred from current protocol alone.

The campaign execution.json has `joint_production_ready:false`, `purpose:transformer_only_three_seeds`, mode='exploratory', user-requested standalone before joint pilot. These are the completed three-seed research runs; do not discard them as timing-only, but do not claim a shared joint timing gate was passed without further evidence.

## Primary results (means +/- sample SD over independent training seeds)

Source `transformer/evaluation/runs/proposal_21/report.json` -> aggregates filtered campaign=mb16, view=best_validation; exact per-seed source tasks/mb16/SIZE/seedSEED/best_validation/summary.json -> metrics.

| Size | Test PPL | BLiMP % | CG nats | Natural Copy100 | Random Copy100 | same-target k128 | distant shuffle128 |
|---|---|---|---|---|---|---|---|
| S | 40.309228 +/-1.093326 | 59.581592 +/-0.435395 | .638982 +/-.025048 | 1.297941 +/-.209442 | 3.396749 +/-1.334285 | .093440 +/-.008287 | .103993 +/-.012908 |
| M | 30.753446 +/-.126013 | 61.493532 +/-.648360 | .691759 +/-.009725 | 1.553315 +/-.024800 | 6.181442 +/-.360983 | .098366 +/-.001827 | .124594 +/-.001079 |
| L | 24.697929 +/-.184643 | 63.969652 +/-.814497 | .713313 +/-.007464 | 1.626212 +/-.034853 | 6.620686 +/-1.053910 | .097441 +/-.001290 | .110837 +/-.004127 |

NLL means S3.696336806/M3.425996436/L3.206700750 (mean PPL is arithmetic mean of run PPL, not exp(mean NLL)). Seed Student-t95% intervals also saved; do not call SD a CI.

Per seed (S13/S37/S73; M13/M37/M73; L13/L37/L73):
- PPL: 41.522542/40.004669/39.400472; 30.859648/30.786483/30.614205; 24.740659/24.857461/24.495667.
- CG: .611562/.644722/.660662; .702081/.690429/.682769; .719773/.715024/.705143.
- Copy100 natural: 1.058314/1.446040/1.389469; 1.539086/1.581952/1.538908; 1.659926/1.628388/1.590322.
- BLiMP fractions: .593955/.600791/.592701; .608881/.621776/.614149; .646537/.641866/.630687.

Secondary mb64 primary-checkpoint means (test PPL/BLiMP fraction/CG/Copy100natural): S40.243286/.595716/.637490/1.309918; M30.744053/.617746/.697353/1.562219; L24.719750/.636781/.712592/1.613646. Results are close at group-mean level but do not claim trajectory identity or double the seed count. Historical seed101 is lower-LR/dropout.1 single seed and should be excluded from main tables.

## Context evidence and interpretation

CG compares mean loss t1--32 minus t385--512 on 542 full test windows. It changes target identity across regions and cannot isolate context benefit. Same-target intervention uses all 69,376 late targets (=542*128), including EOS when present, positions385--512, and scores each with full prefix, k4/32/128 preceding BPE IDs, or a distant shuffle preserving nearest128. Truncated histories prepend BOS and restart position ids at0; shuffle holds length/absolute positions/near128 fixed. Sources Transformer manifests.py, metrics.py and protocol same_target. Full-prefix baseline reuses causally equivalent test forward pass; last-state output-head optimization is tested against unchanged full logits.

Mean k4/k32/k128 gains by size: S1.016712/.333063/.093440; M1.128898/.351720/.098366; L1.199980/.352033/.097441. Positive means intact full history helps. Clear insight: most removable-context gain is local; beyond128 contributes ~.1nat and is nearly flat with size. Increasing CG alone would exaggerate evidence for stronger very-long-context use with scale. Distant shuffle positive too, but its M value exceeds L: no monotonic absolute-scale claim. Shuffling BPE units disrupts words as well as semantic order; truncation also changes absolute positions and recurrent warmup. This supports sensitivity to distant ordered history, not proof that this explains the entire PPL architecture gap.

Cross-implementation same-target target order and shuffle construction appear mathematically identical: T shuffled_history uses seed+[article,start,p+1] for zero-based p; comparison.same_target uses same 1-based position. Any actual manifest equivalence should be checked before claiming exact per-item pairing.

## CRITICAL Copy(L) comparability caveat

Transformer `evaluation/manifests.py` uses ONE sequential PCG64(2718) stream across source and length loops. `comparison/copy_task.py` used by LSTM/KenLM uses independent SeedSequence([2718,source_index,length]) streams. Thus spans are different across architectures despite the shared nominal seed and distribution. The source code proves different generation schemes; inspect actual manifest arrays if accessible for concrete equality. Do not claim exact paired evaluation examples or all shared-manifest hashes match across families. Seed-level model contrast can still be descriptive (same training seed), but uncertainty now includes different finite evaluation samples. Restoring the originally intended paired copy experiment would require re-scoring one common frozen sample; do not silently rerun during report writing.

T Copy gains mean by L25/50/100/200:
- Natural S1.72597/1.57615/1.29794/.99037; M1.89184/1.81268/1.55332/1.25229; L1.80996/1.79883/1.62621/1.37864.
- Random S4.79645/4.26907/3.39675/2.32466; M7.18481/6.98929/6.18144/4.60150; L7.99404/7.53651/6.62069/5.58466.
Higher L is not automatically higher gain. Means decline with span length after25 or50, consistent with greater difficulty over longer exact repetitions; causal attribution needs controls. Random gains are much larger partly because first-span random-ID loss is extremely high (e.g. Lseed13 random100 first13.3028 nats), not directly comparable to natural content prediction.

Natural Copy100 means excluding each span's first4 positions: S1.340575/M1.610167/L1.692599. Independent nonrepeat second-span gains are NEGATIVE: S-.307457/M-.367863/L-.399766. Repeat advantage is not a generic later-position gain, and interior benefit is consistent with memory beyond the 5-gram's4-token horizon. Sources each task summary metrics.copy.natural_100 with keys exclude_first_four_gain and independent_span_gain. No EOS in copy spans, every sequence from BOS/fresh state.

## BLiMP sensitivity, heterogeneity, qualitative illustration

All67,000 pairs,67 paradigms,12groups. Primary sums subword log probabilities INCLUDING EOS, strict good>bad; ties incorrect. Seven official pairs have identical candidates, retained; no overlength sentences. Source task metrics.blimp.counts and evaluator metrics.py. Equal-BPE-length subset=50,019 pairs. Mean equal-length accuracies S58.6864/M60.4923/L63.4599%; length-normalized all-pair scores S57.5781/M58.8458/L61.1861%. Source each task metrics.blimp.sensitivities.[equal_bpe_length_pairs|length_normalized_log_probability].overall.accuracy. These are labelled sensitivities and cannot replace the preregistered primary metric; subset composition differs.

Group-level primary mean percent (S/M/L): determiner-noun agreement73.767/78.579/82.621; irregular forms60.617/75.383/82.833; island effects48.487/47.533/49.562; NPI licensing43.352/44.290/42.243; ellipsis68.417/69.750/60.150. Strong overall scaling coexists with persistent below-chance island/NPI performance and nonmonotonic ellipsis: grammar improvement is heterogeneous, not uniform syntactic mastery. Source primary.groups in task BLiMP summaries (3 seeds averaged).

Traceable failure example, NOT representative random sample: Lseed13 best task metrics.blimp.most_confident_errors_or_smallest_margins and blimp/pairs.jsonl, paradigm matrix_question_npi_licensor_present pair688: grammatical 'Should the school ever collaborate?' vs bad 'The school should ever collaborate.' Model good logP=-100.904161692; bad=-64.3622624813; margin=-36.5418992103 nats; good10 targets/bad9 (includingEOS), wrong under summed AND normalized scores. Deterministic selection: most negative-margin failure, shown as diagnostic only. This demonstrates preference failure; do not assert the cause is grammar alone, token length alone, or the question mark without extra analysis. Cross-architecture disagreement requires LSTM actual pair scores, unavailable in Transformer task alone. No verified repeated-name case selected yet; do not invent one.

## Compute, validation and inference

RTX3090 24GiB, FP32, no autocast/TF32; inference batch16; at most2 independent one-GPU Slurm tasks, not data parallel. Source task hardware and evaluation launchers. Primary T mean recorded training-step seconds S1818.528/M3615.064/L8119.381 =.5051/1.0042/2.2554GPUh per run; peak allocated training memory2.6909/3.4709/5.3574GiB. All9 primary T recorded training-step sum40,658.921sec=11.2941GPUh. These training-step totals exclude validation/checkpoint overhead; not wall time and not entire campaign cost including tuning/sensitivity/evaluation.

Analytic dense training-matmul FLOPs per run S1.3335461137e16/M3.0629887300e16/L8.7513963713e16; total primary9=3.9444200164e17. Source task resources. Formula in runner.py: 3*processed_windows*[layers*(24*T*d^2+4*T^2*d)+2*T*d*V], T512, multiply-add2; backward approximated2xforward, dense T² convention. Excludes norm/nonlinearities/embeddings/optimizer. Do not present as measured FLOPs; comparing another architecture's FLOPs requires the same counting convention.

`completion_audit.json` PASS is saved audit dated2026-11-19, 42 views and105 unique evidence/checkpoint files verified, covers hashes/runtime/targets/raw-loss summaries/pair identity/all copy counts/empty stderr. This is historical audit evidence, not a fresh audit run during this memo. Source unit tests include causal future/padding invariance, tying/counts/finite gradients, last-state/full-logit equivalence, shuffled suffix/target exclusion, natural span boundaries, valid-target denominators, bootstrap/Holm synthetic examples, checkpoint overlap handling and tiny CPU end-to-end resume. Presence of tests does not certify current execution; do not claim fresh test passes.

## Remaining checks before strong final conclusions

Architecture H1/H2/H3 require LSTM result summaries and paired seed contrasts. Match quality only using retained/scored best/final checkpoints at1% validation-PPL tolerance; training log points are not recoverable weights. Compute comparisons need compatible scope. Formal TOST precision/power is not established just by67k pairs; training-seed n3 dominates model-generalization inference. Do not claim equivalence from p>.05. No formal pipeline integrity/review stages were run in this evidence task.
