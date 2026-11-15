# Transformer language model

This folder contains the causal Transformer baseline for **Context, Not Grammar?**:
the model, training command, GPU setup script, and tests. Run all commands below
from the repository root. Shared rules are in
[`comparison/protocol.yaml`](../comparison/protocol.yaml).

**Status, verified 2026-11-15:** S/M/L exploratory training at 250M prediction
tokens and the microbatch-16 timing pilot are complete. The joint LSTM/Transformer
pilot report, hyperparameter tuning, production runs and final evaluation remain
pending. The pilot results below are ready to share with the LSTM owner.

## Completed training

**S, M, and L exploratory training is complete.** Each model finished on
2026-11-11 with seed 101, **250,000,000 prediction tokens**, and **8,087 optimizer
updates**. Results were checked against saved summaries and validation logs on
2026-11-13.

| Size | Final validation NLL | Final validation perplexity | Local run summary |
|---|---:|---:|---|
| S | 4.110006 | **60.95** | [summary.json](runs/s/exploratory_250M_seed101/summary.json) |
| M | 3.773657 | **43.54** | [summary.json](runs/m/exploratory_250M_seed101/summary.json) |
| L | 3.453102 | **31.60** | [summary.json](runs/l/exploratory_250M_seed101/summary.json) |

NLL is mean negative log probability per prediction target, in natural-log units;
perplexity is `exp(NLL)`. Lower is better. These are **final-checkpoint validation
results**, read from `metrics.jsonl`. The `best_validation_nll` field in
`summary.json` describes the best checkpoint, which can differ from the final one.

These are single-seed exploratory results. Production comparisons with seeds
13, 37, and 73 and final test/context/BLiMP evaluations remain pending.
Raw logs and checkpoints live under `transformer/runs/`, which is ignored by Git;
a fresh clone needs a separate copy. See [saved files](#saved-files-and-resume).

### Training settings and measurements

The completed runs share these settings, recorded in `run.json`:

| Setting | Value |
|---|---|
| Runtime | PyTorch 2.7.1+cu126; CUDA runtime 12.6 |
| Learning rate / dropout | 0.0003 / 0.1 |
| Precision | float32 |
| Effective batch | 64 windows per optimizer update |
| Microbatch | 2 windows; 32 microbatches per full update |
| Validation set | 255,099 targets in 529 windows |

Epoch-end and budget-end updates can use smaller batches. Measurements below
come from each run's `summary.json`.

| Size | Prediction tokens/s | Training seconds | Peak CUDA allocation (bytes) | Final validation seconds | Resume checkpoint (bytes) |
|---|---:|---:|---:|---:|---:|
| S | 56,818.05 | 4,400.01 | 529,134,592 | 1.265 | 89,898,315 |
| M | 37,332.10 | 6,696.65 | 779,211,776 | 1.839 | 205,735,691 |
| L | 16,153.18 | 15,476.82 | 1,495,474,688 | 4.959 | 602,571,979 |

- **Throughput:** consumed prediction tokens / accumulated training-step seconds.
  It excludes validation and checkpoint saving, so it is not end-to-end throughput
  or a separate benchmark with a fixed warm-up period.
- **Memory:** PyTorch `max_memory_allocated`, rather than reserved memory or total
  GPU usage. The counter resets at process start/resume; it does not establish
  the peak of an earlier interrupted process.
- **Validation and checkpoint size:** the last full validation pass and the size
  of `latest.pt`, including optimizer and random-generator state.
- **Hardware provenance:** the runs record PyTorch/CUDA versions, but no GPU model
  or UUID. The [microbatch-16 pilot](#completed-microbatch-16-pilot) below records
  hardware identity for its own measurements.

## Completed microbatch-16 pilot

**All three sizes completed successfully on 2026-11-14.** Slurm job **2557**
finished with exit code `0` in **6 minutes 29 seconds**, running S → M → L
sequentially on one GPU. Artifacts were verified on 2026-11-15.

| Setting | Value |
|---|---|
| Hardware | NVIDIA GeForce RTX 3090; `vishc-server-1` |
| Runtime | Python 3.12.9; PyTorch 2.7.1+cu126; CUDA runtime 12.6; driver 595.91.07 |
| CPU allocation | 4 threads |
| Prediction-token budget | 5,000,000 per size; 162 optimizer updates |
| Microbatch / accumulation | **16 windows × 4 = 64 windows per full update** |
| Precision / seed | float32 / 101 |
| Learning rate / dropout | 0.0003 / 0.1; pilot settings, not selected by tuning |
| Validation | All 255,099 targets in 529 windows |

| Size | Prediction tokens/s | Peak CUDA allocation (GiB) | Final validation (s) | Resume checkpoint (MB) | Final validation NLL |
|---|---:|---:|---:|---:|---:|
| S | **157,637.18** | 2.71 | 0.593 | 89.90 | 6.833965 |
| M | **67,579.13** | 3.51 | 1.422 | 205.74 | 6.608971 |
| L | **24,843.05** | 5.44 | 3.909 | 602.57 | 6.501252 |

GiB means bytes / 2³⁰; MB means bytes / 10⁶. Exact byte counts are retained in
the [pilot report](runs/timing_mb16_job2557/transformer_pilot.json).

- **Throughput:** after excluding the first 10 warm-up updates and the final
  budget update, each size contributes **151 full updates / 4,672,306 prediction
  tokens**. The rate is total valid targets / total trainer step seconds; it
  excludes validation, checkpoint I/O and logging. Epoch-partial updates would
  also be excluded. This uses the existing trainer timer, not separate CUDA events.
- **Memory and checkpoint:** peak allocated CUDA memory covers the whole run,
  including warm-up and validation; it is not reserved memory or total GPU use.
  The checkpoint is `latest.pt`, including optimizer and random-generator state.
- **Checks:** all models reached the exact token budget. Input/code/artifact
  hashes matched; validation coverage was unchanged; saved model and optimizer
  tensors were finite. Slurm stderr was empty.
- **Interpretation:** all three models ran at microbatch 16 within the measured
  memory allocation. Short-pilot NLL does not establish converged quality or
  equivalence with microbatch 2. The older 250M-token runs have a different run
  length, warm-up treatment and incomplete hardware provenance, so these tables
  are not a controlled measurement of speedup from changing microbatch size.

**Share with the LSTM owner:**
[`transformer_pilot.json`](runs/timing_mb16_job2557/transformer_pilot.json).
It contains the three required Transformer measurements, runtime/GPU identity,
input and code hashes, per-update throughput statistics and subprocess wall times.
The GPU UUID is `e8e3c43f-2603-7f2a-487c-5843b6551c20`; the protocol SHA-256 is
`f4775f24968c8d28e68db9a203ef1ab7e6d092efcadef529e87d9c91ca5fb688`.

Supporting evidence is under [`runs/timing_mb16_job2557/`](runs/timing_mb16_job2557/):
`s/`, `m/`, `l/` hold the trainer artifacts; each size also has a `.log` and
`_command.json`. Slurm [stdout](runs/slurm/pilot_2557.out) and
[stderr](runs/slurm/pilot_2557.err) are retained. These local artifacts
are ignored by Git; share them separately when another checkout needs them.

The report has **`status: "TRANSFORMER_ONLY"` and `production_ready: false`**.
It still needs LSTM S/M/L evidence bound to the same protocol/inputs and batch
configuration before a joint pilot can unlock production.

## Model architecture

[`model.py`](model.py) implements a decoder-only Transformer with causal
self-attention, learned absolute positions, pre-layer normalization, GELU
feed-forward blocks, residual connections, and tied input/output token embeddings.
The feed-forward width is four times the model width.

| Size | Blocks | Width | Heads | Token-embedding parameters | Non-embedding parameters | Total parameters |
|---|---:|---:|---:|---:|---:|---:|
| S | 4 | 256 | 4 | 4,194,304 | 3,290,624 | 7,484,928 |
| M | 6 | 384 | 6 | 6,291,456 | 10,844,160 | 17,135,616 |
| L | 8 | 640 | 10 | 10,485,760 | 39,717,120 | 50,202,880 |

These are exact counts for a 16,384-token vocabulary and 512 learned positions;
the proposal's 7.3M/16.9M/49.8M figures are approximate. Tied token/output weights
are counted once. Learned positions count as non-embedding parameters for the
protocol's 5% Transformer/LSTM matching check.

Matrix weights start at normal(0, 0.02), normalization scales at one, and biases
at zero. The padding embedding starts at zero and remains trainable through the
tied output head.

The model accepts shared `input_ids`, `attention_mask`, and `position_ids` fields
and returns logits shaped `[batch, time, 16384]`. Each independent window starts
with BOS and positions restart at zero, with at most 512 input positions. The
attention mask describes a nonempty, right-padded prefix. Causal attention blocks
future tokens; the loss mask excludes padding targets.

## GPU setup

Use Linux, Python 3.12, an NVIDIA GPU, and the prepared shared inputs:

- Token arrays and windows: `../dataset/processed/tokens/`.
- Trained tokenizer: `tokenizer/artifacts/`.

See the [dataset guide](../dataset/README.md) to prepare these inputs. All nine
array hashes and the tokenizer hash match
[`dataset/expected_outputs.lock.json`](../dataset/expected_outputs.lock.json),
confirmed on 2026-11-13. Follow the
[cross-machine verification instructions](../dataset/README.md#verify-data-across-machines)
when using another copy.

```bash
bash transformer/setup_gpu.sh
source .venv/bin/activate
python -m pytest transformer/tests -q
```

The shared `requirements.txt` pins CPU PyTorch 2.7.1+cpu. For GPU training, use
[`setup_gpu.sh`](setup_gpu.sh), which installs PyTorch 2.7.1 from the official
CUDA 12.6 wheel index and resolves dependencies from PyPI. Its final check
requires CUDA to be available; run it where an allocated GPU is visible.

| Setup override | Default | Purpose |
|---|---|---|
| `TRANSFORMER_PYTHON` | `python3.12` | Python executable for creating `.venv/` |
| `PYTORCH_CUDA_INDEX` | `https://download.pytorch.org/whl/cu126` | Official wheel index compatible with the GPU driver |
| `TRANSFORMER_PIP_TIMEOUT` | `600` | pip socket timeout, in seconds |
| `TRANSFORMER_PIP_RETRIES` | `10` | pip connection retries |

Completed downloads are cached in `.venv/pip-cache`. The script retries each
failed pip operation up to three times, reusing cached downloads.

## Run an exploratory timing/training trial on one GPU

This example runs a size-S trial with **1M prediction tokens**, using the tested
microbatch-16 configuration, inside an allocated GPU session:

```bash
python -m transformer.train --mode exploratory --size s \
  --token-dir ../dataset/processed/tokens \
  --budget-tokens 1000000 --microbatch-windows 16 \
  --learning-rate 0.0003 --dropout 0.1 --seed 101 \
  --run-dir transformer/runs/s/exploratory_1M_mb16_seed101
```

Choose `--size s`, `m`, or `l` and a separate run directory for each experiment.
The run reports actual memory and throughput in `summary.json`. Training uses
float32, 64 windows per full effective batch, exact prediction-token accounting,
validation-only checkpoint selection, and a token-based cosine learning-rate schedule.

On a managed cluster, run inside a scheduler GPU allocation and preserve its
`CUDA_VISIBLE_DEVICES` mapping. Allow only one process to write to each run
directory. A single Transformer trial does not complete the six-model timing
pilot required for production.

## Run the microbatch-16 timing pilot

[`timing_pilot.py`](timing_pilot.py) reproduces the configuration in the
[completed pilot](#completed-microbatch-16-pilot) using the unchanged trainer,
with a fresh process per size. Its **5M-token/size budget is for timing**, separate
from tuning and production. It requires at least 20 measured full updates after
warm-up and exports the four required measurements for each size.

The script verifies the current protocol and inputs and creates a new output
directory with its own `protocol.prepared.lock.json`. The original shared
preparation lock predates the `same_target` evaluation amendment; it remains
unchanged with the older runs. Existing output directories are never overwritten.

On this server, submit from the repository root:

```bash
mkdir -p transformer/runs/slurm
pilot_job_id=$(env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  sbatch --parsable transformer/timing_pilot.sbatch)
echo "Pilot job ID: $pilot_job_id"
```

The job requests one RTX 3090 and runs all sizes on that allocation. It keeps
Slurm's GPU mapping and writes logs to `transformer/runs/slurm/pilot_JOBID.out`
and `.err`. Its one-hour limit is a cap, not a runtime estimate. It does not
automatically resume/requeue; a failed run retains its evidence. Submit a new
job to retry. Optional Python arguments can follow the batch-script filename,
for example `--budget-tokens 10000000`.

Monitor the job; once it starts, follow its log:

```bash
squeue -j "$pilot_job_id"
tail -f "transformer/runs/slurm/pilot_${pilot_job_id}.out"
```

After completion, inspect the report:

```bash
cat "transformer/runs/timing_mb16_job${pilot_job_id}/transformer_pilot.json"
```

Inside an existing GPU allocation, the equivalent command is:

```bash
.venv/bin/python -u -m transformer.timing_pilot \
  --output-dir transformer/runs/timing_mb16_trial01
```

The report also retains the original trainer rate including warm-up. Subprocess
wall times include setup, data verification, validation and saves. Use matching
timing definitions for the LSTM report. If the shared training loop changes,
remeasure the implementation that will be used for production.

## Saved files and resume

| File | Contents |
|---|---|
| `run.json` | Run settings, runtime versions, and input/code hashes |
| `metrics.jsonl` | Training and validation events |
| `summary.json` | Completion status, best validation score, parameter counts, and measured timing/memory |
| `best.pt` | Model weights at the lowest validation NLL |
| `final.pt` | Model weights at the final token budget |
| `latest.pt` | Resume state: model, optimizer, random generators, epoch, window cursor, token count, and schedule progress |

To resume an interrupted run, repeat its original command with `--resume`.
`latest.pt` is required. The trainer checks data, protocol, model/training code,
run settings, and PyTorch/CUDA versions against the saved identity. Changed
settings or versions require a new run directory. Without `--resume`, a nonempty
run directory is rejected.

## Production runs

**Transformer timing is complete; the shared production gate is still closed.**
The pilot-local snapshot is preparation-only. The older shared preparation lock
has `production_ready: false`, `pilot: null`, and an outdated protocol hash.

The next steps are:

1. Combine Transformer measurements with LSTM S/M/L evidence using the same
   protocol/input hashes, float32, microbatch 16 and accumulation 4. Confirm
   consistent timing definitions and record hardware for both architectures.
2. Agree exact budgets in the joint report. **250M prediction tokens per
   production run and 50M per tuning trial are proposed settings** until frozen
   in the production lock. The 50M tuning budget counts tokens, not parameters.
3. Create and verify a new production-ready snapshot from that joint report:

   ```bash
   python -m comparison.experiment_settings --pilot-report /path/to/pilot.json \
     --snapshot-file comparison/protocol.production.lock.json
   python -m comparison.experiment_settings --check --require-ready \
     --snapshot-file comparison/protocol.production.lock.json
   ```

4. Run the eight size-S tuning trials **per architecture**, seed 101: learning
   rates `[0.0001, 0.0003, 0.001, 0.003]` × dropouts `[0.0, 0.1]`, with the same
   token budget. Select each architecture's winner by validation NLL and reuse
   it for M/L and the deep-LSTM control without additional tuning. The current
   trainer has no tuning mode; use explicit exploratory trial budgets to carry
   out this grid. A production-ready snapshot alone does not prove tuning is done.
5. Run Transformer S/M/L × seeds **13, 37, 73** with `--mode production` and
   `--snapshot-file`, using the selected hyperparameters. The trainer reads the
   exact budget/microbatch from the lock. Test/context/copy/BLiMP evaluation
   follows validation-only checkpoint selection.

Neither the completed 250M exploratory runs nor the 5M timing pilots replace
these production runs. Test data remains reserved for final evaluation.

See the [experiment guide](../comparison/README.md#why-production-training-is-not-ready-yet)
for the pilot-report format and readiness checks.
