#!/usr/bin/env bash
# Linux server setup. Set PYTORCH_CUDA_INDEX to cu118/cu128 if required by the driver.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TRANSFORMER_PYTHON:-python3.12}"
cuda_index="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"
"$python_bin" -m venv "$project_root/.venv"
python="$project_root/.venv/bin/python"
requirements="$(mktemp)"
trap 'rm -f "$requirements"' EXIT
grep -vE '^(--extra-index-url|torch==)' "$project_root/requirements.txt" > "$requirements"
"$python" -m pip install -r "$requirements"
"$python" -m pip install 'torch==2.7.1' --index-url "$cuda_index"
"$python" -m pip check
"$python" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA PyTorch or GPU driver unavailable"'
