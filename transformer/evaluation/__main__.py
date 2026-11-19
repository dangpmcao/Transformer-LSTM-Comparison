"""python -m transformer.evaluation {prepare,run,collect,smoke}."""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')

from pathlib import Path
_local_cache = Path(__file__).resolve().parent / '.cache'
os.environ.setdefault('CUDA_CACHE_PATH', str(_local_cache / 'cuda'))
os.environ.setdefault('TRITON_CACHE_DIR', str(_local_cache / 'triton'))
os.environ.setdefault('TORCH_HOME', str(_local_cache / 'torch'))

import argparse
from pathlib import Path
import json

from .common import inside, read, write
from .runner import DEFAULT_CONFIG, DEFAULT_OUTPUT, prepare, run, verify, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    for name in ['run', 'collect', 'smoke']:
        p = commands.add_parser(name)
        p.add_argument('--manifest', type=Path, default=DEFAULT_OUTPUT / 'manifest.json')
        if name in ['run', 'smoke']:
            p.add_argument('--task-index', type=int, default=0)
            p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
        else:
            p.add_argument('--comparison-bundle', type=Path)
            p.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.config, args.output_dir)
        print(json.dumps({'status': 'prepared', 'tasks': len(result['tasks']),
                          'output_dir': str(args.output_dir), 'coverage': result['coverage']}, indent=2))
    elif args.command == 'run':
        result = run(args.manifest, args.task_index, args.device)
        print(json.dumps({'status': result['status'], 'checkpoint': result['checkpoint']['path']}))
    elif args.command == 'collect':
        from .report import collect
        result = collect(args.manifest, args.comparison_bundle, not args.no_plots)
        print(json.dumps({'status': result['status'], 'completed': result['completed_checkpoint_tasks'],
                          'expected': result['expected_checkpoint_tasks']}, indent=2))
    else:
        from .smoke import smoke
        print(json.dumps(smoke(args.manifest, args.task_index, args.device), indent=2))


if __name__ == '__main__':
    main()
