#!/usr/bin/env bash
#SBATCH --job-name=transformer-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-1
#SBATCH --array=0-41%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=transformer/evaluation/slurm/eval_%A_%a.out
#SBATCH --error=transformer/evaluation/slurm/eval_%A_%a.err
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Usage: sbatch transformer/evaluate.sbatch MANIFEST [--smoke]' >&2
  exit 1
fi
cd "${SLURM_SUBMIT_DIR:?}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_CACHE_PATH="$PWD/transformer/evaluation/.cache/cuda"
export TRITON_CACHE_DIR="$PWD/transformer/evaluation/.cache/triton"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MPLCONFIGDIR="$PWD/transformer/evaluation/.matplotlib"
if [[ "${2:-}" == '--smoke' ]]; then
  exec srun --ntasks=1 .venv/bin/python -u -m transformer.evaluation smoke \
    --manifest "$1" --task-index "${SLURM_ARRAY_TASK_ID:-0}" --device cuda
fi
exec srun --ntasks=1 .venv/bin/python -u -m transformer.evaluation run \
  --manifest "$1" --task-index "${SLURM_ARRAY_TASK_ID:?}" --device cuda
