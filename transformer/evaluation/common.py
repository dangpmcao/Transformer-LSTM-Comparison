"""Contained, atomic artifacts and frozen evidence checks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def inside(path: str | Path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"Evaluation output must be inside {ROOT}: {path}")
    return path


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path: Path, value: Any) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def record(path: Path) -> dict:
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def verify_record(item: dict) -> None:
    p = Path(item['path'])
    if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
        raise ValueError(f'Frozen evidence changed: {p}')


def code_records() -> list[dict]:
    files = sorted((ROOT / 'evaluation').glob('*.py'))
    files += [ROOT / 'evaluate.sbatch', ROOT / 'model.py', PROJECT / 'comparison/blimp_scorer.py',
              PROJECT / 'dataset/windows.py', PROJECT / 'dataset/encode.py',
              PROJECT / 'tokenizer/core.py', PROJECT / 'dataset/prepare.py']
    return [record(p) for p in files]
