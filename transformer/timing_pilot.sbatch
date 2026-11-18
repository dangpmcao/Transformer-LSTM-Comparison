#!/usr/bin/env bash
# Submit from NLP-Project after creating transformer/runs/slurm/.
#SBATCH --job-name=transformer-pilot16
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=transformer/runs/slurm/pilot_%j.out
#SBATCH --error=transformer/runs/slurm/pilot_%j.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'A Slurm GPU allocation is required. Submit this file with sbatch.' >&2
  exit 1
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the NLP-Project root}"
if [[ ! -x .venv/bin/python || ! -f transformer/timing_pilot.py ]]; then
  echo 'Submit from NLP-Project with its prepared .venv.' >&2
  exit 1
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
echo "job=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# All three sizes run sequentially on this allocation. Keep Slurm's GPU mapping.
exec srun --ntasks=1 .venv/bin/python -u -m transformer.timing_pilot \
  --output-dir "transformer/runs/timing_mb16_job${SLURM_JOB_ID}" "$@"
