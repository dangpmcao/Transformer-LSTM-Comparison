#!/usr/bin/env bash
# Submit from the repository root after creating transformer/runs/slurm/.
#SBATCH --job-name=transformer-explore
#SBATCH --partition=gpu
#SBATCH --nodelist=vishc-server-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --array=0-2%2
#SBATCH --output=transformer/runs/slurm/%A_%a.out
#SBATCH --error=transformer/runs/slurm/%A_%a.err

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo 'Submit with sbatch; a Slurm GPU allocation is required.' >&2
  exit 1
fi
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if [[ ! -x .venv/bin/python || ! -f transformer/train.py ]]; then
  echo 'Submit from NLP-Project with its prepared .venv.' >&2
  exit 1
fi
case "${SLURM_ARRAY_TASK_ID:-}" in
  0) size=s ;;
  1) size=l ;;
  2) size=m ;;
  *) echo 'Expected array task 0 (S), 1 (L), or 2 (M).' >&2; exit 1 ;;
esac

# Hold the lock outside the run folder so a new run starts in an empty folder.
mkdir -p transformer/runs/slurm
exec 9>"transformer/runs/slurm/${size}_exploratory_250M_seed101.lock"
if ! flock --nonblock 9; then
  echo "Another Slurm launcher already owns the $size run." >&2
  exit 1
fi
run_dir="transformer/runs/${size}/exploratory_250M_seed101"
resume=()
if [[ -f "$run_dir/latest.pt" ]]; then
  resume=(--resume)
fi
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID size=$size node=$(hostname)"
echo "Slurm GPUs=${SLURM_JOB_GPUS:-unset} CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES resume=${resume[*]:-no}"

# Preserve Slurm's device mapping; never select physical GPU IDs here.
# More frequent checkpoints limit work lost on a later interruption.
exec srun --ntasks=1 .venv/bin/python -u -m transformer.train \
  --mode exploratory --size "$size" --device cuda \
  --token-dir /mnt/disk1/backup_user/dang.cpm/NLP/dataset/processed/tokens \
  --budget-tokens 250000000 --microbatch-windows 2 \
  --learning-rate 0.0003 --dropout 0.1 --seed 101 \
  --save-every-updates 25 --run-dir "$run_dir" "${resume[@]}"
