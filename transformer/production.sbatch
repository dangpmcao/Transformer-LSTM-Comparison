#!/usr/bin/env bash
# Nine independent model/seed runs, one GPU each, at most two running at once.
#SBATCH --job-name=transformer-production
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-1
#SBATCH --array=0-8%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=transformer/runs/slurm/production_%A_%a.out
#SBATCH --error=transformer/runs/slurm/production_%A_%a.err

set -euo pipefail
if [[ $# -ne 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Usage from NLP-Project: sbatch transformer/production.sbatch CAMPAIGN_DIR' >&2
  exit 1
fi
cd "${SLURM_SUBMIT_DIR:?}"
if [[ ! -x .venv/bin/python || ! -f transformer/production.py ]]; then
  echo 'Submit from NLP-Project with its prepared CUDA .venv.' >&2
  exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
echo "job=$SLURM_JOB_ID run=$SLURM_ARRAY_TASK_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# The Python runner skips completed runs, locks each writer and resumes latest.pt.
exec srun --ntasks=1 .venv/bin/python -u -m transformer.production run \
  --campaign-dir "$1" --run-index "$SLURM_ARRAY_TASK_ID"
