"""Persist observed environment capabilities, including explicit GPU failures."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .common import DEFAULT_DATA_ROOT, PROJECT_ROOT, utc_now, write_json


def command_result(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def inspect_environment(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    packages = {}
    for name in ["torch", "datasets", "huggingface-hub", "tokenizers", "numpy", "scipy",
                 "pandas", "matplotlib", "pytest", "pyarrow", "requests", "kenlm"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    import torch

    imports = {name: importlib.import_module(name).__name__ for name in
               ["datasets", "tokenizers", "numpy", "scipy", "pandas", "matplotlib", "pyarrow"]}
    imports["kenlm"] = importlib.import_module("kenlm").__name__ if importlib.util.find_spec("kenlm") else "not_installed"

    torch.set_num_threads(2)
    # Stage 1 CPU smoke test; no training or GPU allocation.
    matrix = torch.ones((2, 3), dtype=torch.float32)
    cpu_smoke = bool(torch.equal(matrix @ matrix.T, torch.full((2, 2), 3.0)))
    disk = shutil.disk_usage(root)
    tools = PROJECT_ROOT / ".tools/kenlm/build/bin"
    result = {
        "observed_at": utc_now(), "python": sys.version, "executable": sys.executable,
        "prefix": sys.prefix, "base_prefix": sys.base_prefix, "isolated_venv": sys.prefix != sys.base_prefix,
        "platform": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
        "packages": packages, "import_smoke": imports, "cpu_tensor_smoke": cpu_smoke,
        "torch": {"version": torch.__version__, "cuda_build": torch.version.cuda,
                  "cuda_available": torch.cuda.is_available(), "cudnn": torch.backends.cudnn.version()},
        "nvidia_smi": command_result(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]),
        "memory": command_result(["free", "-b"]),
        "disk": {"path": str(root.resolve()), "total_bytes": disk.total, "free_bytes": disk.free},
        "git_revision": command_result(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"]),
        "git_status": command_result(["git", "-C", str(PROJECT_ROOT), "status", "--short"]),
        "compiler": command_result(["g++", "--version"]),
        "kenlm_binaries": {name: str(tools/name) if (tools/name).exists() else None
                           for name in ["lmplz", "build_binary"]},
        "purpose": "CPU-only Stage 1 environment; CUDA training environment must be validated before Stage 5.",
    }
    write_json(root / "environment.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    print(json.dumps(inspect_environment(args.data_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
