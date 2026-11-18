#!/usr/bin/env bash
# Nine short timing tasks; at most two allocated GPUs across the whole array.
#SBATCH --job-name=transformer-batch-bench
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-1
#SBATCH --array=0-8%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=transformer/runs/slurm/batch_bench_%A_%a.out
#SBATCH --error=transformer/runs/slurm/batch_bench_%A_%a.err

set -euo pipefail
if [[ $# -ne 1 || -z "${SLURM_JOB_ID:-}" || -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Usage from NLP-Project: sbatch transformer/batch_benchmark.sbatch BENCHMARK_DIR' >&2
  exit 1
fi
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
exec srun --ntasks=1 .venv/bin/python -u -m transformer.batch_benchmark run \
  --benchmark-dir "$1" --task-index "$SLURM_ARRAY_TASK_ID"
