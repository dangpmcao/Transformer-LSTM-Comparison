#!/usr/bin/env bash
# Prepare, validate S/L on allocated GPUs, then score 42 checkpoint views.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
OUTPUT="${1:-$PWD/transformer/evaluation/runs/proposal_21}"
.venv/bin/python -m transformer.evaluation prepare --output-dir "$OUTPUT"
mkdir -p transformer/evaluation/slurm
smoke_job_id=$(env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  sbatch --parsable --job-name=transformer-eval-smoke --array=0,12%2 --time=00:10:00 \
  transformer/evaluate.sbatch "$OUTPUT/manifest.json" --smoke)
printf 'GPU smoke array: %s\n' "$smoke_job_id"
evaluation_job_id=$(env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  sbatch --parsable --dependency="afterok:${smoke_job_id%%;*}" \
  transformer/evaluate.sbatch "$OUTPUT/manifest.json")
printf 'Full evaluation array: %s (starts after both GPU smokes pass)\n' "$evaluation_job_id"
