"""Small real-checkpoint/data inference check, explicitly separate from final results."""
from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np
import torch

from .common import inside, read, write, sha
from .manifests import shuffled_history
from .runner import verify, load_model
from .scoring import Scorer


def smoke(manifest: Path, index: int, device: str) -> dict:
    manifest = inside(manifest)
    frozen = verify(manifest)
    task = frozen['tasks'][index]
    verify(manifest, task)
    if device == 'cuda' and (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')
                             or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError('GPU smoke requires one Slurm-assigned GPU')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    batch_size = frozen['settings']['inference_batch_size']
    scorer = Scorer(load_model(task, device), device, batch_size)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    shared = manifest.parent / 'shared'
    with np.load(shared / 'test.npz', allow_pickle=False) as data:
        w = int(data['same_window'][0]);p = int(data['same_position'][0])
        targets = data['targets'][w].copy();row = data['index'][w].copy()
    whole = scorer.windows([targets.tolist()] * batch_size)[0]
    full = scorer.last([[1, *targets[:p].tolist()]], [int(targets[p])])[0]
    assert abs(full - whole[p]) < 5e-4, (full, whole[p])
    assert abs(scorer.windows([targets.tolist()] * batch_size)[0][p] - whole[p]) < 1e-6
    histories = [[1, *targets[p-k:p].tolist()] for k in [4, 32, 128]]
    histories.append(shuffled_history(targets, p, int(row[0]), int(row[1])).tolist())
    assert np.isfinite(scorer.last(histories, [int(targets[p])] * 4)).all()
    with np.load(shared / 'copy_natural_25.npz', allow_pickle=False) as data:
        first, other = data['first'][:2], data['independent'][:2]
    assert all(len(x) == 50 for x in scorer.windows(np.concatenate([first, first], axis=1).tolist()))
    assert all(len(x) == 50 for x in scorer.windows(np.concatenate([first, other], axis=1).tolist()))
    blimp = read(shared / 'blimp.json')
    assert all(np.isfinite(x).all() for x in scorer.windows(blimp['windows'][:8]))
    result = {'status': 'SMOKE_PASS_NOT_FINAL_EVALUATION', 'task_index': index,
              'checkpoint_sha256': task['checkpoint']['sha256'], 'device': device,
              'wall_seconds': time.perf_counter() - started, 'full_target_nll': float(full),
              'max_positions': scorer.model.config.max_positions, 'inference_batch_size': batch_size,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
              'checks': ['512-target test window', 'final-hidden-state vs full-logit score',
                         'deterministic repeat', '4/32/128/full/shuffled histories',
                         'natural copy and independent control', 'BLiMP token windows']}
    suffix = os.environ.get('SLURM_JOB_ID', 'local')
    write(manifest.parent / 'smoke' / f'{device}_task{index}_{suffix}.json', result)
    return result
