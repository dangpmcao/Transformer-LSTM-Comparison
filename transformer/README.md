# Transformer language model

Causal Transformer baseline for **Context, Not Grammar?** This guide covers
results, training, checkpoint recovery, and reproduction of the tuning and timing
studies. Shared rules are in
[`comparison/protocol.yaml`](../comparison/protocol.yaml).

Results: [results.md](results.md).

Run all commands from the repository root:

```bash
cd /mnt/disk1/backup_user/dang.cpm/NLP/NLP-Project
```

## Status

**Verified 2026-11-16 against local run artifacts.**

| Stage | Status | Outcome |
|---|---|---|
| Initial S/M/L training | Complete | Seed 101, 250M prediction tokens per size |
| Size-S tuning | All 12 trials complete | Selected learning rate **0.001**, dropout **0.0** |
| Batch benchmark | All 9 tasks complete | Selected microbatch **64**, accumulation **1** |
| Three-seed Transformer campaign | In progress; **2/9 runs complete** | L seeds 13 and 37 finished; see results below |
| Joint Transformer/LSTM production | Awaiting joint pilot | Requires both architectures' measurements and agreed budgets |
| Final evaluation | Pending | Test perplexity, context, copy, and BLiMP |

The current campaign runs S/M/L with seeds **13, 37, 73**, at **250M prediction
tokens per run**. It uses the explicit Transformer-only option. Its saved trainer
mode is `exploratory`; the campaign records its three-seed purpose separately in
[`execution.json`](runs/transformer_only_250M_3seed/execution.json). Joint
production readiness remains false. Check LSTM budgets and settings before
claiming a matched comparison between architectures.

Budgets count valid next-token prediction targets, including EOS and excluding
BOS and padding. A **microbatch** is processed at once on one GPU; **gradient
accumulation** combines microbatches before an optimizer update. For example,
16 windows × 4 microbatches gives an effective batch of 64 windows.

**Go to:** [Results](#completed-training) ·
[Run and monitor](#run-all-three-seeds-on-two-gpus) ·
[GPU setup](#gpu-setup) · [Model](#model-architecture) ·
[Saved files](#saved-files-and-resume) · [Tuning](#tune-size-s-on-two-gpus) ·
[Timing measurements](#batch-benchmark)

## Completed training

NLL is mean negative log probability per prediction target, in natural-log units.
Perplexity (PPL) is `exp(NLL)`; lower is better. All scores below use the validation
set: **255,099 targets in 529 windows**. Best-checkpoint scores come from
`summary.json`; final-budget scores come from the last validation event in
`metrics.jsonl`.

### Current three-seed campaign

Each completed run consumed exactly **250,000,000 prediction tokens** in
**8,087 optimizer updates**. Settings: learning rate **0.001**, dropout **0.0**,
float32, microbatch **64 × accumulation 1**.

| Size | Seed | Best validation NLL | Best validation PPL | Final-budget PPL | Summary |
|---|---:|---:|---:|---:|---|
| L | 13 | 3.197021 | **24.46** | 24.48 | [summary.json](runs/transformer_only_250M_3seed/l/seed13/summary.json) |
| L | 37 | 3.207560 | **24.72** | 24.75 | [summary.json](runs/transformer_only_250M_3seed/l/seed37/summary.json) |

These are partial campaign results. Report comparisons across seeds after all
nine runs and final evaluation are complete. Use the monitoring command below
for the latest completion count.

### Initial exploratory runs

Completed on **2026-11-11**, with seed **101**, 250M prediction tokens and 8,087
updates per size. These used learning rate **0.0003**, dropout **0.1**, float32,
and microbatch **2 × accumulation 32 = 64 windows per full update**. Epoch-end
and budget-end updates can use smaller batches.

| Size | Final validation NLL | Final validation PPL | Summary |
|---|---:|---:|---|
| S | 4.110006 | **60.95** | [summary.json](runs/s/exploratory_250M_seed101/summary.json) |
| M | 3.773657 | **43.54** | [summary.json](runs/m/exploratory_250M_seed101/summary.json) |
| L | 3.453102 | **31.60** | [summary.json](runs/l/exploratory_250M_seed101/summary.json) |

These are single-seed, final-checkpoint validation results. M/L best-checkpoint
scores differ. Settings changed between these runs and the current campaign,
so the tables do not isolate the effect of any one change.

<details>
<summary>Initial-run timing and memory measurements</summary>

| Size | Prediction tokens/s | Training seconds | Peak CUDA allocation (bytes) | Final validation (s) | Resume checkpoint (bytes) |
|---|---:|---:|---:|---:|---:|
| S | 56,818.05 | 4,400.01 | 529,134,592 | 1.265 | 89,898,315 |
| M | 37,332.10 | 6,696.65 | 779,211,776 | 1.839 | 205,735,691 |
| L | 16,153.18 | 15,476.82 | 1,495,474,688 | 4.959 | 602,571,979 |

Throughput is consumed tokens / accumulated training-step seconds, including
warm-up and excluding validation and saves. Peak memory is PyTorch allocated
memory; its counter resets on process start/resume. It does not measure reserved
memory, total device use, or the peak of an earlier interrupted process.
Checkpoint size is for `latest.pt`, including optimizer and random-generator
state. Runtime was PyTorch **2.7.1+cu126**, CUDA **12.6**; GPU model and UUID were
not recorded for these initial runs.

</details>

## Run all three seeds on two GPUs

### Monitor the current campaign

The current campaign was submitted as Slurm array **2615**. Inspect its saved
results and the live scheduler separately:

```bash
.venv/bin/python -m transformer.production collect \
  --campaign-dir transformer/runs/transformer_only_250M_3seed
squeue -u "$USER"
```

`collect` verifies the campaign and completed artifacts. Its `running` status
comes from saved attempt records; use Slurm to confirm a job is still active.
Logs are `transformer/runs/slurm/production_JOBID_TASKID.out` and `.err`.

### Start or resume a Transformer-only campaign

The command below prepares and submits all nine runs. For the current campaign,
use it for recovery **after the prior array has stopped**:

```bash
bash transformer/run_production.sh --standalone
```

| Setting | Value |
|---|---|
| Runs | S/M/L × seeds 13, 37, 73 |
| Task order | 0–2: L; 3–5: M; 6–8: S; seeds 13/37/73 within each size |
| Budget | Exactly 250M prediction tokens per run |
| Hyperparameters | Size-S tuning winner: LR 0.001, dropout 0.0 |
| Batch | 64 windows per update; microbatch 64 × accumulation 1 |
| Training | Deterministic float32, AdamW, token-based cosine schedule |
| Parallelism | Array `0-8%2`; one RTX 3090 per task, at most two tasks concurrently |
| Resources per task | 4 CPU threads, 16 GiB host RAM, 24-hour time limit |
| Output | `transformer/runs/transformer_only_250M_3seed/{s,m,l}/seed{13,37,73}/` |

The launcher freezes settings, inputs, code, runtime, and the tuning winner
before submitting. Each task checks them again, locks its output directory,
skips completed runs, and resumes interrupted runs from `latest.pt`.
See [recovery requirements](#saved-files-and-resume).

Two GPUs run independent experiments; they do not split one model's batch.
Microbatch changes retain an effective batch of 64, normalized loss, clipping,
and update count, subject to float32 summation rounding. CPU tests cover padding,
partial epochs, and a truncated final token budget. The launcher uses the existing
trainer without adding AMP, TF32, or changing the model or optimizer.

Slurm assigns the GPUs. Preserve its `CUDA_VISIBLE_DEVICES` mapping. The two-GPU
allowance applies across your jobs; wait for an active array to finish before
submitting another training or benchmark array. Time limits are caps, not runtime
estimates.

### Joint production after the shared pilot

This path requires a joint report with Transformer **and** LSTM S/M/L timing and
memory measurements, current protocol/input hashes, agreed exact production and
tuning budgets, and a common microbatch/accumulation setting. The launcher requires
the tuning budget to match the completed Transformer study: **50M tokens/trial**.
See the [joint report format](../comparison/README.md#why-production-training-is-not-ready-yet).

Once that report is available at `comparison/pilot.production.json`:

```bash
.venv/bin/python -m comparison.experiment_settings \
  --pilot-report comparison/pilot.production.json \
  --snapshot-file comparison/protocol.production.lock.json
bash transformer/run_production.sh comparison/protocol.production.lock.json
```

The joint path requires a production-ready lock and takes its exact budget and
batch settings from that lock. Its default output is
`transformer/runs/production_3seed/`; inspect it with
`.venv/bin/python -m transformer.production collect`. Preview the proposed
nine-run matrix without submitting via
`.venv/bin/python -m transformer.production plan`.

Both launcher paths accept an optional campaign directory as their second
argument. Reuse the same path and snapshot for recovery. Test/context/copy/BLiMP
evaluation is a separate step after checkpoint selection by validation.

## GPU setup

Use Linux, Python 3.12, an NVIDIA GPU, and these shared inputs:

| Input | Path from repository root |
|---|---|
| Token arrays and windows | `../dataset/processed/tokens/` |
| Trained tokenizer | `tokenizer/artifacts/` |

The [dataset guide](../dataset/README.md) explains preparation and
[cross-machine verification](../dataset/README.md#verify-data-across-machines).
Expected hashes are in
[`dataset/expected_outputs.lock.json`](../dataset/expected_outputs.lock.json).
The nine array hashes and tokenizer hash were confirmed on 2026-11-13.

Run setup where an allocated GPU is visible:

```bash
bash transformer/setup_gpu.sh
.venv/bin/python -m pytest transformer/tests -q
```

The repository's `requirements.txt` pins CPU PyTorch. The GPU setup script
installs **PyTorch 2.7.1 with CUDA 12.6**, resolves dependencies from PyPI, and
checks that CUDA is available. Commands in this guide use `.venv/bin/python`.

| Setup override | Default | Purpose |
|---|---|---|
| `TRANSFORMER_PYTHON` | `python3.12` | Python used to create `.venv/` |
| `PYTORCH_CUDA_INDEX` | `https://download.pytorch.org/whl/cu126` | Official CUDA wheel index |
| `TRANSFORMER_PIP_TIMEOUT` | `600` | pip socket timeout in seconds |
| `TRANSFORMER_PIP_RETRIES` | `10` | pip connection retries |

Downloads are cached in `.venv/pip-cache`; failed pip operations are retried up
to three times using cached downloads.

## Run an exploratory timing/training trial on one GPU

For a short trial inside an existing scheduler GPU allocation:

```bash
.venv/bin/python -m transformer.train --mode exploratory --size s \
  --token-dir ../dataset/processed/tokens \
  --budget-tokens 1000000 --microbatch-windows 16 \
  --learning-rate 0.0003 --dropout 0.1 --seed 101 \
  --run-dir transformer/runs/s/exploratory_1M_mb16_seed101
```

Choose size `s`, `m`, or `l`, and a fresh run directory per experiment. This
example uses **1M prediction tokens**, float32, microbatch 16 × accumulation 4,
exact token accounting, and validation-only checkpoint selection. A short trial
does not complete the shared six-model timing pilot. Measured timing and memory
are saved in `summary.json`.

## Model architecture

[`model.py`](model.py) implements a decoder-only Transformer: causal attention,
learned absolute positions, pre-layer normalization, GELU feed-forward blocks
with width 4× the model width, residual connections, and tied input/output token
embeddings.

| Size | Blocks | Width | Heads | Token-embedding parameters | Non-embedding parameters | Total parameters |
|---|---:|---:|---:|---:|---:|---:|
| S | 4 | 256 | 4 | 4,194,304 | 3,290,624 | 7,484,928 |
| M | 6 | 384 | 6 | 6,291,456 | 10,844,160 | 17,135,616 |
| L | 8 | 640 | 10 | 10,485,760 | 39,717,120 | 50,202,880 |

Counts use **16,384 vocabulary tokens** and **512 learned positions**. Tied
token/output weights count once; learned positions count as non-embedding
parameters for the protocol's 5% Transformer/LSTM matching check. The proposal's
7.3M/16.9M/49.8M totals are approximate.

Matrix weights start at normal(0, 0.02), normalization scales at one, and biases
at zero. The padding embedding starts at zero and remains trainable through the
tied output head.

Inputs are `input_ids`, `attention_mask`, and `position_ids`; output logits have
shape `[batch, time, 16384]`. Each independent window starts with BOS and resets
positions to zero, with at most 512 input positions. The attention mask describes
a nonempty, right-padded prefix; causal attention blocks future tokens, and the
loss mask excludes padding targets.

## Saved files and resume

Each training run saves:

| File | Contents |
|---|---|
| `run.json` | Settings, runtime versions, and input/code hashes |
| `metrics.jsonl` | Training and validation events |
| `summary.json` | Completion status, best validation score, parameter counts, timing, and memory |
| `best.pt` | Model weights at the lowest validation NLL |
| `final.pt` | Model weights at the final token budget |
| `latest.pt` | Model, optimizer, random generators, epoch, window cursor, token count, and schedule progress |

**For a direct trainer command**, repeat the original command with `--resume`.
**For a campaign or tuning study**, resubmit its launcher after the prior jobs
stop; it adds resume automatically and skips completed runs. Campaign checkpoints
are saved every 100 updates and at epoch/end. Per-run locks prevent duplicate
writers; `attempts/` records commands, Slurm/device mappings, exits, and retries.

Resume requires `latest.pt` and matching data, protocol, model/training code,
settings, and PyTorch/CUDA versions. Changed settings need a new run directory.
Without resume, the trainer rejects a nonempty directory. If a job fails before
its first checkpoint, investigate and use a new study/campaign directory,
retaining the failed evidence. Campaign jobs enable scheduler requeue, which
does not guarantee retries for time limits or training errors.

**`transformer/runs/` is ignored by Git.** Preserve and share these artifacts
separately; a fresh clone does not include logs, reports, or checkpoints.

## Tune size S on two GPUs

The completed study uses **seed 101**, **50M prediction tokens/trial**, float32,
and **microbatch 16 × accumulation 4**. The grid is:

- Learning rate: `[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]`.
- Dropout: `[0.0, 0.1]`, giving **12 trials**.

The grid was extended symmetrically on 2026-11-16 after the LSTM owner's boundary
optimum. All trials completed; **trial 04** won with LR **0.001**, dropout **0.0**,
best validation NLL **4.546444 / PPL 94.30** at 50M tokens. See its
[summary](runs/s/tuning_50M_seed101_6lr/trial_04/summary.json).
The selected LR/dropout transfers to M/L without extra tuning.

Selection uses lowest best-validation NLL, then grid order for ties. It waits
until all trials have completed or recorded failure; failed trials cannot win.
Investigate infrastructure failures before accepting selection. The production
launcher requires every trial to complete successfully. Test and BLiMP scores
are not used. Compare the LSTM tuning budget before making equal-budget claims.

Inspect the existing study:

```bash
.venv/bin/python -m transformer.tuning collect \
  --study-dir transformer/runs/s/tuning_50M_seed101_6lr
```

To reproduce in a new directory:

```bash
TUNE_ROOT="$PWD/transformer/runs/s/tuning_50M_seed101_repeat01"
.venv/bin/python -m transformer.tuning prepare \
  --study-dir "$TUNE_ROOT" --budget-tokens 50000000 --microbatch-windows 16
mkdir -p transformer/runs/slurm
tune_job_id=$(env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  sbatch --parsable transformer/tune_s.sbatch "$TUNE_ROOT")
squeue -j "$tune_job_id"
```

The array is `0-11%2`, one RTX 3090 per trial, with a two-hour limit per task.
Grid order is LR ascending, then dropout 0.0/0.1: tasks 0/1 use `1e-4`, tasks
10/11 use `3e-2`. Logs use `tune_s_JOBID_TASKID.out` and `.err` under
`transformer/runs/slurm/`; outputs are `trial_00` through `trial_11`.

Collect results after completion:

```bash
.venv/bin/python -m transformer.tuning collect --study-dir "$TUNE_ROOT" \
  > "$TUNE_ROOT/selection.json"
```

Resubmit the same study to recover, or select tasks using
`sbatch --array=4,9%2 transformer/tune_s.sbatch "$TUNE_ROOT"` with the same
`env -u` prefix shown above.

Preparation freezes the protocol, inputs, budget, and code. Repeating `prepare`
verifies an existing study; changed settings require a new directory. Tuning
wraps the existing trainer in `exploratory` mode and records its purpose in
`study.json`. It does not create a production-ready lock.

## Batch benchmark

Completed on **2026-11-16**, Slurm array **2606**, with all nine tasks successful.
Each size tested microbatches **16/32/64** for **2M tokens/task**, on RTX 3090,
float32, LR **0.001**, dropout **0.0**.

| Size | Tokens/s, batch 16 | Tokens/s, batch 32 | Tokens/s, batch 64 | Peak allocated GiB, batch 64 |
|---|---:|---:|---:|---:|
| S | 162,081 | 164,341 | 161,124 | 10.18 |
| M | 79,456 | 84,594 | 85,852 | 12.82 |
| L | 32,692 | 34,526 | 34,575 | 18.76 |

The selected common microbatch is **64 × accumulation 1**. Selection minimizes
summed S/M/L subprocess wall seconds per million tokens:
**69.02 (64), 69.12 (32), 71.29 (16)**, requiring at least 10% headroom based on
peak allocated GPU memory. Batches 32 and 64 were nearly tied; S alone was faster
with 32. This short study identifies the fastest observed common setting, without
establishing optimal speed or convergence equivalence.

Training rates exclude the first 10 warm-up updates, partial/final updates,
validation, saves, and logging; at least 20 full measured updates are required.
Subprocess wall time includes setup, input verification, validation, and saves.
Peak allocation is not reserved memory or total device use. GiB = bytes / 2³⁰;
MB = bytes / 10⁶.

See [full results](runs/batch_benchmark_20261116/performance.json) and the
[Transformer pilot contribution](runs/batch_benchmark_20261116/transformer_pilot.json)
for exact measurements, GPU/runtime identity, and input/code/artifact hashes.
The contribution has status `TRANSFORMER_ONLY`; add LSTM evidence at the agreed
setting before creating a joint report. The
[planned matrix](runs/production_3seed.plan.json) records the proposed nine runs.

<details>
<summary>Reproduce the batch benchmark</summary>

Use a new output directory:

```bash
BENCH="$PWD/transformer/runs/batch_benchmark_repeat01"
.venv/bin/python -m transformer.batch_benchmark prepare --benchmark-dir "$BENCH" \
  --tuning-study transformer/runs/s/tuning_50M_seed101_6lr
mkdir -p transformer/runs/slurm
env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  sbatch --parsable transformer/batch_benchmark.sbatch "$BENCH"
```

After all tasks finish:

```bash
.venv/bin/python -m transformer.batch_benchmark collect --benchmark-dir "$BENCH" \
  > "$BENCH/performance.json"
.venv/bin/python -m transformer.production export-pilot --benchmark-dir "$BENCH" \
  > "$BENCH/transformer_pilot.json"
```

</details>

## Completed microbatch-16 pilot

This earlier timing pilot completed on **2026-11-14**, Slurm job **2557**, running
S → M → L on one RTX 3090 in **6 min 29 s** (exit code 0). It used **5M tokens/size**
(162 updates), seed 101, float32, LR 0.0003, dropout 0.1, and **microbatch 16 ×
accumulation 4**. Runtime: Python 3.12.9, PyTorch 2.7.1+cu126, CUDA 12.6, driver
595.91.07, four CPU threads on `vishc-server-1`.

| Size | Prediction tokens/s | Peak allocated GiB | Final validation (s) | Resume checkpoint (MB) | Final validation NLL |
|---|---:|---:|---:|---:|---:|
| S | 157,637.18 | 2.71 | 0.593 | 89.90 | 6.833965 |
| M | 67,579.13 | 3.51 | 1.422 | 205.74 | 6.608971 |
| L | 24,843.05 | 5.44 | 3.909 | 602.57 | 6.501252 |

Rates use **151 full updates / 4,672,306 targets per size**, excluding 10 warm-up
updates and the final budget update, with the trainer's existing timer. Memory
covers the whole run, including validation. All sizes reached the exact budget
and validated on all 255,099 targets. The recorded audit confirmed hashes,
finite saved model/optimizer tensors, and empty Slurm stderr.

The [pilot report](runs/timing_mb16_job2557/transformer_pilot.json) retains exact
bytes, GPU UUID, protocol/input hashes, and source measurements. Supporting logs
and commands are in [the run directory](runs/timing_mb16_job2557/) and Slurm
[stdout](runs/slurm/pilot_2557.out)/[stderr](runs/slurm/pilot_2557.err).
Its status is `TRANSFORMER_ONLY`, with `production_ready: false`. Pilot NLL is
a short-run diagnostic. Different run lengths, settings, and warm-up treatment
prevent a controlled speedup comparison with the initial 250M-token runs.

<details>
<summary>Run the microbatch-16 timing pilot again</summary>

```bash
mkdir -p transformer/runs/slurm
pilot_job_id=$(env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  sbatch --parsable transformer/timing_pilot.sbatch)
squeue -j "$pilot_job_id"
tail -f "transformer/runs/slurm/pilot_${pilot_job_id}.out"
```

After completion, read
`transformer/runs/timing_mb16_job${pilot_job_id}/transformer_pilot.json`.
Optional Python arguments follow the batch-script filename, such as
`--budget-tokens 10000000`. Inside an existing GPU allocation, use:

```bash
.venv/bin/python -u -m transformer.timing_pilot \
  --output-dir transformer/runs/timing_mb16_repeat01
```

Each size runs in a fresh process. The pilot creates a new directory and a
current input-bound `protocol.prepared.lock.json`, retaining older locks and
artifacts. The original shared preparation lock predates the `same_target`
amendment and is not suitable for current joint production. Existing output
directories are rejected. This job has a one-hour limit and no automatic
resume/requeue; keep failed evidence and submit a new job to retry.

Match timing definitions with the LSTM report. If the training loop changes,
remeasure the implementation used for production.

</details>
