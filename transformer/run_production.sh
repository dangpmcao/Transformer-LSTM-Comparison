#!/usr/bin/env bash
# Freeze and validate all nine runs BEFORE requesting GPU time.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash transformer/run_production.sh {--standalone|PRODUCTION_SNAPSHOT} [CAMPAIGN_DIR]' >&2
  exit 1
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ "$1" == '--standalone' ]]; then
  campaign="${2:-$PWD/transformer/runs/transformer_only_250M_3seed}"
  .venv/bin/python -m transformer.production prepare-standalone \
    --budget-tokens 250000000 --microbatch-windows 64 --campaign-dir "$campaign"
else
  campaign="${2:-$PWD/transformer/runs/production_3seed}"
  .venv/bin/python -m transformer.production prepare \
    --snapshot-file "$1" --campaign-dir "$campaign"
fi
mkdir -p transformer/runs/slurm
env -u CUDA_VISIBLE_DEVICES -u GPU_DEVICE_ORDINAL \
  sbatch --parsable transformer/production.sbatch "$campaign"
