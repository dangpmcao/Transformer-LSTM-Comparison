# Transformer proposal evaluation

All new code, configuration, manifests, temporary files, Slurm logs and results
live under `transformer/`. The shared data, tokenizer, proposal and comparison
modules are read-only inputs. Existing training code, checkpoints and locks are
not changed. Commands run from `/mnt/disk1/backup_user/dang.cpm/NLP/NLP-Project`.

## Frozen matrix

| Campaign | Runs | Role |
|---|---:|---|
| `mb16` | S/M/L × seeds 13/37/73 = 9 | Primary, training microbatch 16 × accumulation 4 |
| `mb64` | S/M/L × seeds 13/37/73 = 9 | Batch sensitivity, training microbatch 64 × accumulation 1 |
| `initial` | S/M/L × seed 101 = 3 | Historical single-seed results; excluded from three-seed inference |

Every run has two separately labelled views: `best_validation` (`best.pt`, selected
only by WikiText validation) and `final_budget` (`final.pt`). There are **42
checkpoint tasks**, not 42 independent training seeds. Inference uses float32,
dropout disabled, TF32 disabled, fresh independent BOS windows and batch size 16.
The inference batch does not change the saved training configuration.

## Measures and proposal coverage

| Measure | Implementation and saved evidence |
|---|---|
| Test NLL/PPL | All 291,710 test targets, including EOS and short windows; summed loss / summed valid targets, followed by exp |
| Position loss / context gain, H1/H3 | Position curve on all 542 full windows; mean positions 1–32 minus 385–512; short-window exclusions reported |
| Copy gain, H1/H3 | 1,000 samples per source/length, lengths 25/50/100/200, seed 2718; natural content and uniform IDs 4–16383; natural Copy(100) is primary |
| Copy controls | Independently sampled nonrepeat second span; also exclude the first four targets of each span; every per-target loss is saved |
| BLiMP, H2 | All 67,000 pairs; summed subword log probability including EOS, strict comparison, ties incorrect; overall/12 groups/67 paradigms; equal-BPE-length and length-normalized sensitivities |
| TA same-target ablation | All 69,376 targets at positions 385–512; full/4/32/128-token histories and distant shuffle preserving the nearest 128 tokens; paired per-target scores and exclusions saved |
| Uncertainty | Article-cluster bootstrap for WikiText, binary-pair bootstrap for BLiMP, 10,000 replicates; separate sample SD and Student-t 95% intervals across training seeds |
| Resource accounting | Exact model/embedding/non-embedding counts and original training times; clearly labelled analytic training matmul FLOPs estimate; evaluation time/hardware/peak allocated GPU bytes |
| Scaling | Individual seeds, means and position curves across S/M/L, exported as PNG/SVG and CSV/JSON |
| Checkpoint quality matching | Uses only available, actually scored best/final weights and validation PPL, tolerance 1%; reports no overlap when no retained pair matches; never treats validation-log rows as recoverable weights |
| Architecture hypothesis tests | Optional imported LSTM bundle: paired-seed H1 (Holm family 6), H2 ±2 percentage-point TOST (family 3), H3 L-minus-S architecture contrasts (family 2) |

The first/late-position metric is the proposal's original endpoint. Same-target
truncation/shuffling is a supporting intervention from the current protocol,
not a replacement for that endpoint. BPE shuffling also changes distant word
forms. Full-prefix scores reuse the equivalent causal test forward pass.
Truncated/shuffled scores apply the output head only to the last valid hidden
state; this is position-wise equivalent to the unchanged model forward, up to
float32 rounding, and is checked against full logits in CPU/GPU smoke tests.

FLOPs use `3 × processed_windows × [layers × (24Td² + 4T²d) + 2TdV]`, with
T=512 and multiply-add counted as two operations. This is a **dense-matmul
estimate**, with a dense T² attention convention and backward approximated as
2× forward. It excludes embedding lookup, normalization, nonlinearities and
optimizer operations. It is not a measured FLOPs counter. For historical resume rollbacks, the
retained trajectory and all logged executed updates (including discarded work)
are counted separately; raw training logs are never rewritten. Timing excludes or
includes overhead exactly as labelled in the source training summaries.

Natural copy bootstrap clusters by the first source article; random sequences
are separate evaluation examples. These are conditional evaluation uncertainties,
not additional training replicates. Hypothesis tests use the three paired
training seeds only. The `mb64` reruns are never pooled with `mb16` as six seeds.

## Prepare and launch

```bash
cd /mnt/disk1/backup_user/dang.cpm/NLP/NLP-Project
export PYTHONDONTWRITEBYTECODE=1
.venv/bin/python -m transformer.evaluation prepare
bash transformer/evaluate.sh
```

Preparation verifies all 21 completed runs, input/code/runtime identities and
checkpoint hashes, then persists the ordered targets, copy spans and encoded
BLiMP pairs **before any scoring**. Repeating preparation verifies the existing
configuration; different settings/code/runtime require a new output directory.
The default directory is `transformer/evaluation/runs/proposal_21/`.

`evaluate.sh` first submits short S/L GPU smoke tasks (`0,12%2`). The full
array depends on **both succeeding** (`afterok`), so a GPU-validation failure
prevents the full scoring stage from starting. Both job IDs are printed.

`evaluate.sbatch` uses array `0-41%2`: at most **two independent one-GPU
jobs**, each requesting a 24 GB RTX 3090, four CPUs, 24 GiB host RAM, and a
24-hour cap on `vishc-server-1`. It keeps Slurm's device mapping. There is no
model/data parallel training. A job may wait for the scheduler to allocate GPUs.

To prepare a different output directory without changing any source artifact:

```bash
bash transformer/evaluate.sh "$PWD/transformer/evaluation/runs/proposal_21_repeat01"
```

## Smoke checks and progress

```bash
.venv/bin/python -m transformer.evaluation smoke --device cpu --task-index 0
mkdir -p transformer/evaluation/slurm
# Two small jobs: S and L, both at the configured inference batch 16.
env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  sbatch --array=0,12%2 transformer/evaluate.sbatch \
  "$PWD/transformer/evaluation/runs/proposal_21/manifest.json" --smoke
squeue -u "$USER"
.venv/bin/python -m transformer.evaluation collect
```

Smokes write `SMOKE_PASS_NOT_FINAL_EVALUATION` artifacts under `smoke/`; they do
not produce final metric results. Slurm logs are under
`transformer/evaluation/slurm/eval_JOBID_TASKID.out` and `.err`. Each task has a
`status.json` heartbeat. Saved `running` status alone is not proof a job is live;
check Slurm. The collector reports completed versus missing checkpoint tasks.

Completed metric stages are checksum-verified and skipped. Same-target work is
saved every 2,048 targets and resumes from those chunks. A writer lock prevents
duplicate writers per checkpoint. Resubmit the same array after previous jobs
stop to recover incomplete work. Scheduler requeue is enabled but does not
promise recovery from timeouts or training/inference errors. Do not change the
frozen code/configuration while an evaluation is running.

## Results

`tasks/CAMPAIGN/SIZE/seedSEED/VIEW/` contains per-target test losses, BLiMP pair
scores, repeated/control copy losses, same-target condition scores, stage
summaries and a checksum-bound final `summary.json`. The final result includes
the checkpoint hash and original training identity/resources.

`collect` writes `report.json`, `run_results.json`, `results.csv`, scaling/position
figures and `comparison_contract.json`, all within the evaluation directory.
A complete checkpoint task count means **Transformer scoring is complete**;
it does not claim the full architecture comparison is complete.

## LSTM, deep-LSTM and KenLM results

These 21 saved runs are all Transformers. No LSTM or KenLM weights/results are
invented or trained by this configuration. The proposal's architecture gaps,
equivalence claims and baseline controls need actual collaborator outputs.
The collector explicitly lists those missing inputs.

The LSTM owner should reuse the exact frozen `shared/` manifests. An optional
JSON bundle for `collect --comparison-bundle PATH` has:

```json
{
  "architecture": "lstm",
  "evaluation_protocol_sha256": "from comparison_contract.json",
  "tokenizer_sha256": "from comparison_contract.json",
  "shared_manifest_hashes": {"test.npz": "...", "blimp.json": "...", "copy_natural_25.npz": "..."},
  "results": [
    {
      "size": "s", "seed": 13, "view": "best_validation",
      "checkpoint_sha256": "actual checkpoint hash", "validation_nll": 4.0,
      "metrics": {"test": {}, "copy": {}, "blimp": {}, "same_target": {}}
    }
  ]
}
```

The example is a schema sketch, not measured evidence. Include **all** shared
manifest hashes and all three seeds at S/M/L. Each `metrics` object uses the same
schema as a Transformer task summary. Include final-budget views for quality
matching, optional `m_deep` results, and an optional `pairs_file` record
(`path`, `bytes`, `sha256`) per checkpoint for cross-model disagreement cases. Supply `parameters.non_embedding` per
LSTM result for the ±5% matching audit. An optional `baselines.kenlm` entry
contains an actual `model_sha256` and measured `metrics`; it is never synthesized.
No H1/H2/H3 result is produced from incomplete or unpaired seed sets.

Confirmatory tests use the predeclared primary campaign's best-validation view.
Final-budget, microbatch-sensitivity and historical results remain separately
labelled. Additional baseline/depth evidence is required before claiming that
all proposal controls and fairness requirements have been satisfied.
