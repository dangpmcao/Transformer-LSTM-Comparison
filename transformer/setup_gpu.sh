#!/usr/bin/env bash
# Linux server setup. Set PYTORCH_CUDA_INDEX to cu118/cu128 if required by the driver.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${TRANSFORMER_PYTHON:-python3.12}"
cuda_index="${PYTORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu126}"
"$python_bin" -m venv "$project_root/.venv"
python="$project_root/.venv/bin/python"
work_dir="$(mktemp -d)"
trap 'rm -rf -- "$work_dir"' EXIT
requirements="$work_dir/requirements.txt"
pip_options=(--timeout "${TRANSFORMER_PIP_TIMEOUT:-600}"
  --retries "${TRANSFORMER_PIP_RETRIES:-10}"
  --cache-dir "$project_root/.venv/pip-cache")

# Older pip versions do not retry interrupted response bodies. Retry the whole
# operation too; the project-local cache preserves completed downloads.
retry_pip() {
  local attempt
  for attempt in 1 2 3; do
    if "$python" -m pip "$@" "${pip_options[@]}"; then
      return 0
    fi
    echo "pip attempt $attempt of 3 failed: $*" >&2
  done
  return 1
}

grep -vE '^(--extra-index-url|torch==)' "$project_root/requirements.txt" > "$requirements"
retry_pip install -r "$requirements" --index-url https://pypi.org/simple
# Keep the selected official CUDA wheel, but resolve its pinned dependencies
# from PyPI instead of the NVIDIA download links in the CUDA wheel index.
retry_pip download 'torch==2.7.1' --no-deps --only-binary=:all: \
  --index-url "$cuda_index" --dest "$work_dir"
torch_wheels=("$work_dir"/torch-*.whl)
if [[ ${#torch_wheels[@]} -ne 1 || ! -f "${torch_wheels[0]}" ]]; then
  echo 'Expected exactly one downloaded PyTorch wheel.' >&2
  exit 1
fi
retry_pip install "${torch_wheels[0]}" --index-url https://pypi.org/simple
"$python" -m pip check
"$python" -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available()); assert torch.cuda.is_available(), "CUDA PyTorch or GPU driver unavailable"'
