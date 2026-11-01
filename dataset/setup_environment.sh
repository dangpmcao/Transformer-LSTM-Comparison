#!/usr/bin/env bash
# Stage 1 preparation only. The CPU wheel intentionally does not enable CUDA training.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
stage1_python="${STAGE1_PYTHON:-python3.12}"
if ! "$stage1_python" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3, 12))'; then
  echo 'Python 3.12 is required. Set STAGE1_PYTHON to your Python 3.12 executable.' >&2
  exit 1
fi
"$stage1_python" -m venv "$project_root/.venv"
"$project_root/.venv/bin/python" -m pip install --no-cache-dir --timeout 30 -r "$project_root/requirements.txt"
"$project_root/.venv/bin/python" -m pip check
