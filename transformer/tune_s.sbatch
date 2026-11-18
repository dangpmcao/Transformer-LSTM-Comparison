#!/usr/bin/env bash
# Submit from NLP-Project. Each trial gets one GPU; at most two run together.
#SBATCH --job-name=transformer-s-tune
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-1
#SBATCH --array=0-11%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=transformer/runs/slurm/tune_s_%A_%a.out
#SBATCH --error=transformer/runs/slurm/tune_s_%A_%a.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Submit with sbatch: a Slurm GPU array allocation is required.' >&2
  exit 1
fi
if [[ $# -ne 1 ]]; then
  echo 'Usage: sbatch transformer/tune_s.sbatch STUDY_DIR' >&2
  exit 1
fi
cd "${SLURM_SUBMIT_DIR:?Submit from NLP-Project}"
if [[ ! -x .venv/bin/python || ! -f transformer/tuning.py ]]; then
  echo 'Submit from NLP-Project with its prepared CUDA .venv.' >&2
  exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
echo "job=$SLURM_JOB_ID trial=$SLURM_ARRAY_TASK_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# Keep the device mapping assigned by Slurm. Independent trials need no DDP.
exec srun --ntasks=1 .venv/bin/python -u -m transformer.tuning run \
  --study-dir "$1" --trial-index "$SLURM_ARRAY_TASK_ID"
