"""Freeze 21 runs / 42 checkpoint tasks and run immutable, resumable evaluation."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch
import yaml

from comparison.protocol import load_protocol
from dataset.common import DEFAULT_DATA_ROOT, SOURCE_LOCK
from dataset.windows import load_windows
from tokenizer.core import load_tokenizer
from transformer.model import TransformerConfig, TransformerLM
from . import manifests, metrics
from .common import ROOT, PROJECT, inside, read, sha, write, record, verify_record, code_records

DEFAULT_CONFIG = Path(__file__).with_name('config.yaml')
DEFAULT_OUTPUT = ROOT / 'evaluation/runs/proposal_21'
VIEWS = {'best_validation': 'best.pt', 'final_budget': 'final.pt'}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime() -> dict:
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'numpy': np.__version__}


def configuration(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if config['schema_version'] != 1 or config['checkpoint_views'] != list(VIEWS):
        raise ValueError('Require both frozen checkpoint views')
    if config['metrics'] != ['test', 'blimp', 'copy', 'same_target']:
        raise ValueError('All declared proposal metrics are required')
    for name in ['inference_batch_size', 'target_chunk_size', 'copy_examples',
                 'bootstrap_replicates', 'case_examples']:
        if type(config[name]) is not int or config[name] <= 0:
            raise ValueError(f'Invalid positive setting: {name}')
    if (config['copy_lengths'] != [25, 50, 100, 200] or config['copy_examples'] != 1000
            or config['copy_seed'] != 2718 or config['shuffle_seed'] != 1618
            or config['bootstrap_seed'] != 31415 or config['bootstrap_replicates'] != 10000):
        raise ValueError('Sample coverage/seeds/uncertainty must follow the locked protocol')
    expected = [('mb16', 'primary', [13, 37, 73]), ('mb64', 'batch_sensitivity', [13, 37, 73]),
                ('initial', 'historical_single_seed', [101])]
    if [(r['name'], r['role'], r['seeds']) for r in config['campaigns']] != expected:
        raise ValueError('Require the two 9-run campaigns and the initial three runs')
    if config['primary_campaign'] != 'mb16':
        raise ValueError('Primary campaign is fixed before test/BLiMP evaluation')
    return config


def canonical_history(events: list[dict], expected_updates: int) -> tuple[list[dict], list[dict], int]:
    """Retain the final trajectory after checkpoint rollbacks; keep raw logs untouched."""
    updates, validations = {}, {}
    executed = 0
    for event in events:
        number = event['updates']
        if event['event'] == 'update':
            executed += 1
            if updates and number <= max(updates):
                updates = {n: row for n, row in updates.items() if n < number}
                validations = {n: row for n, row in validations.items() if n < number}
            updates[number] = event
        elif event['event'] == 'validation':
            validations[number] = event
    if set(updates) != set(range(1, expected_updates + 1)):
        raise ValueError('Final training trajectory has missing updates')
    return ([validations[n] for n in sorted(validations)],
            [updates[n] for n in sorted(updates)], executed - len(updates))


def prepare(config_path: Path, output: Path) -> dict:
    output = inside(output)
    config = configuration(config_path)
    if output.exists():
        frozen = verify(output / 'manifest.json')
        if frozen['settings'] != config:
            raise ValueError('Existing evaluation has another configuration')
        return frozen
    token_dir = DEFAULT_DATA_ROOT / 'processed/tokens'
    load_windows(token_dir / 'windows', token_dir)
    load_tokenizer(PROJECT / 'tokenizer/artifacts')
    protocol = load_protocol()
    shared_inputs = [PROJECT / 'proposal.pdf', PROJECT / 'comparison/protocol.yaml', SOURCE_LOCK]
    shared_inputs += sorted((PROJECT / 'tokenizer/artifacts').glob('*'))
    shared_inputs += [token_dir / 'manifest.json', token_dir / 'windows/manifest.json',
                      DEFAULT_DATA_ROOT / 'processed/blimp/paradigm_groups.json']
    shared_inputs += sorted(token_dir.glob('*.npy')) + sorted((token_dir / 'windows').glob('*.npy'))
    input_records = [record(p) for p in shared_inputs if p.is_file()]
    tasks = []
    token_sha = sha(token_dir / 'manifest.json')
    window_sha = sha(token_dir / 'windows/manifest.json')
    tokenizer_sha = sha(PROJECT / 'tokenizer/artifacts/tokenizer.json')
    for group in config['campaigns']:
        base = inside(PROJECT / group['root'])
        for size in ['s', 'm', 'l']:
            for seed in group['seeds']:
                d = base / size / (f'seed{seed}' if group['name'] != 'initial' else 'exploratory_250M_seed101')
                summary = read(d / 'summary.json')
                identity = read(d / 'run.json')['identity']
                if (summary['status'] != 'complete' or summary['consumed_tokens'] != 250000000
                        or identity['size'] != size or identity['seed'] != seed
                        or identity['tokenizer_sha256'] != tokenizer_sha
                        or identity['token_manifest_sha256'] != token_sha
                        or identity['window_manifest_sha256'] != window_sha
                        or identity['model_code_sha256'] != sha(ROOT / 'model.py')
                        or identity['train_code_sha256'] != sha(ROOT / 'train.py')):
                    raise ValueError(f'Incompatible or incomplete saved run: {d}')
                config_model = TransformerConfig(**identity['model'])
                config_model.validate()
                events = [json.loads(line) for line in (d / 'metrics.jsonl').read_text().splitlines()]
                validation, updates, discarded = canonical_history(events, summary['updates'])
                if (validation[-1]['consumed_tokens'] != 250000000
                        or min(e['nll'] for e in validation) != summary['best_validation_nll']
                        or sum(e['prediction_tokens'] for e in updates) != 250000000):
                    raise ValueError(f'Summary/log mismatch: {d}')
                windows = sum(e['batch_windows'] for e in updates)
                # Analytic dense matmul estimate: multiply-add = 2 FLOPs;
                # backward approximately 2x forward. Counts dense T*T attention,
                # excludes embeddings/LayerNorm/nonlinearities/optimizer; not a profiler measurement.
                t, width, layers = 512, config_model.width, config_model.layers
                forward = layers * (24 * t * width**2 + 4 * t**2 * width) + 2 * t * width * config_model.vocab_size
                resources = {'parameters': summary['parameters'], 'training_step_seconds': summary['train_seconds'],
                             'training_targets_per_second': summary['prediction_tokens_per_second'],
                             'training_peak_allocated_bytes': summary['peak_cuda_bytes'],
                             'optimizer_updates': summary['updates'], 'processed_windows': windows,
                             'discarded_logged_updates_after_resume': discarded,
                             'estimated_logged_executed_matmul_flops_including_discarded': int(3 * forward * sum(e['batch_windows'] for e in events if e['event'] == 'update')),
                             'estimated_dense_training_matmul_flops': int(3 * forward * windows),
                             'flops_method': '3x forward dense linear+attention matmuls, multiply-add=2; '
                                             'dense T*T attention convention; excludes embedding lookup, '
                                             'norm, nonlinearities and optimizer; analytic estimate, not measured'}
                evidence = [record(d / name) for name in ['run.json', 'summary.json', 'metrics.jsonl']]
                for view, filename in VIEWS.items():
                    tasks.append({'index': len(tasks), 'campaign': group['name'], 'role': group['role'],
                                  'architecture': 'transformer', 'size': size, 'seed': seed, 'view': view,
                                  'directory': f"{group['name']}/{size}/seed{seed}/{view}",
                                  'checkpoint': record(d / filename), 'run_evidence': evidence,
                                  'identity': identity, 'resources': resources,
                                  'validation_nll': summary['best_validation_nll'] if view == 'best_validation' else validation[-1]['nll']})
    if len(tasks) != 42:
        raise ValueError('Expected exactly 21 runs and 42 checkpoint views')
    output.mkdir(parents=True)
    try:
        coverage = manifests.build(output / 'shared', token_dir, PROJECT / 'tokenizer/artifacts', DEFAULT_DATA_ROOT, config)
        shared = [record(p) for p in sorted((output / 'shared').iterdir()) if p.is_file()]
        frozen = {'schema_version': 1, 'created_at': now(), 'settings': config,
                  'runtime': runtime(), 'protocol': protocol, 'proposal': record(PROJECT / 'proposal.pdf'),
                  'evaluation_protocol_sha256': sha(PROJECT / 'comparison/protocol.yaml'),
                  'tokenizer_sha256': tokenizer_sha, 'inputs': input_records, 'shared': shared,
                  'code': code_records(), 'tasks': tasks, 'coverage': coverage,
                  'comparisons_requiring_external_evidence': ['LSTM nine runs', 'M-deep LSTM three runs',
                       'KenLM', 'paired architecture hypothesis tests', 'matched validation PPL overlap']}
        write(output / 'manifest.json', frozen)
        write(output / 'manifest_record.json', record(output / 'manifest.json'))
        write(output / 'protocol.snapshot.json', protocol)
        write(output / 'configuration.json', config)
    except BaseException:
        # Only newly created evaluation outputs are removed; training evidence is never touched.
        shutil.rmtree(output)
        raise
    return frozen


def verify(path: Path, task: dict | None = None) -> dict:
    path = inside(path)
    verify_record(read(path.parent / 'manifest_record.json'))
    frozen = read(path)
    if frozen['runtime'] != runtime():
        raise ValueError('Evaluation runtime changed; use a new output directory')
    if frozen['code'] != code_records():
        raise ValueError('Evaluation code changed; retain outputs and prepare a new evaluation')
    for item in frozen['inputs'] + frozen['shared']:
        verify_record(item)
    if task is not None:
        for item in task['run_evidence'] + [task['checkpoint']]:
            verify_record(item)
    return frozen


def load_model(task: dict, device: str) -> TransformerLM:
    saved = torch.load(task['checkpoint']['path'], map_location='cpu', weights_only=False)
    if saved['identity'] != task['identity']:
        raise ValueError('Checkpoint identity does not match its frozen run')
    expected = task['identity']['budget_tokens'] if task['view'] == 'final_budget' else read(Path(task['run_evidence'][1]['path']))['best_checkpoint_tokens']
    if saved['consumed_tokens'] != expected:
        raise ValueError('Checkpoint has the wrong token budget/selection')
    if task['view'] == 'best_validation' and saved['validation']['nll'] != task['validation_nll']:
        raise ValueError('Checkpoint is not the validation-selected model')
    model = TransformerLM(TransformerConfig(**task['identity']['model']))
    model.load_state_dict(saved['model'], strict=True)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise ValueError('Checkpoint contains nonfinite model weights')
    return model.to(device).eval()


def result_valid(root: Path, identity: dict) -> dict | None:
    path = root / 'summary.json'
    if not path.exists():
        return None
    result = read(path)
    if result['evaluation_identity'] != identity:
        raise ValueError(f'Evaluation identity changed: {root}')
    for artifact in result['artifacts']:
        verify_record(artifact)
    return result


def run(manifest: Path, index: int, device: str) -> dict:
    manifest = inside(manifest)
    frozen = verify(manifest)
    if not 0 <= index < len(frozen['tasks']):
        raise ValueError('Task index outside frozen matrix')
    task = frozen['tasks'][index]
    verify(manifest, task)
    root = inside(manifest.parent / 'tasks' / task['directory'])
    root.mkdir(parents=True, exist_ok=True)
    identity = {'manifest_sha256': sha(manifest), 'task_index': index,
                'checkpoint_sha256': task['checkpoint']['sha256']}
    with (root / 'writer.lock').open('a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('This evaluation already has a writer') from error
        if (root / 'identity.json').exists() and read(root / 'identity.json') != identity:
            raise ValueError('Refusing to reuse results from different settings/code/weights')
        write(root / 'identity.json', identity)
        previous = result_valid(root, identity)
        if previous:
            print(f'Already complete: {task["directory"]}', flush=True)
            return previous
        if device == 'cuda' and (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')
                                 or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError('Evaluation requires exactly one Slurm-assigned visible GPU per task')
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '4')))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        from .scoring import Scorer
        started = time.perf_counter()
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        state = {'status': 'running', 'task': task['directory'], 'started_at': now(),
                 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'device': device}
        last_ping = 0.0
        def progress():
            nonlocal last_ping
            if time.monotonic() - last_ping > 30:
                state['heartbeat_at'] = now()
                write(root / 'status.json', state)
                last_ping = time.monotonic()
        progress()
        scorer = Scorer(load_model(task, device), device,
                        frozen['settings']['inference_batch_size'], progress)
        functions = {'test': lambda out: metrics.test_metrics(scorer, manifest.parent / 'shared', out, frozen['settings']),
                     'blimp': lambda out: metrics.blimp_metrics(scorer, manifest.parent / 'shared', out, frozen['settings']),
                     'copy': lambda out: metrics.copy_metrics(scorer, manifest.parent / 'shared', out, frozen['settings']),
                     'same_target': lambda out: metrics.same_target_metrics(scorer, manifest.parent / 'shared', out,
                                                    frozen['settings'], root / 'test/scores.npz')}
        measured = {}
        try:
            for name, function in functions.items():
                state['metric'] = name
                progress()
                stage = root / name
                previous = result_valid(stage, identity)
                if previous:
                    measured[name] = previous['result']
                    continue
                stage.mkdir(exist_ok=True)
                tick = time.perf_counter()
                value = function(stage)
                artifacts = [record(p) for p in sorted(stage.rglob('*')) if p.is_file() and p.name != 'summary.json']
                write(stage / 'summary.json', {'evaluation_identity': identity, 'result': value,
                      'wall_seconds': time.perf_counter() - tick, 'artifacts': artifacts})
                measured[name] = value
            result = {'status': 'complete', 'evaluation_identity': identity,
                      'campaign': task['campaign'], 'architecture': 'transformer', 'size': task['size'],
                      'seed': task['seed'], 'view': task['view'], 'checkpoint': task['checkpoint'],
                      'validation_nll': task['validation_nll'], 'resources': task['resources'],
                      'metrics': measured, 'finished_at': now(), 'wall_seconds': time.perf_counter() - started,
                      'timing_scope': 'current process attempt; completed stages/chunks may be reused on resume',
                      'hardware': {'host': platform.node(), 'device': device,
                          'gpu_name': torch.cuda.get_device_name() if device == 'cuda' else None,
                          'gpu_total_bytes': torch.cuda.get_device_properties(0).total_memory if device == 'cuda' else None,
                          'peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None},
                      'artifacts': [record(p) for p in sorted(root.rglob('*'))
                                    if p.is_file() and p.name not in ['summary.json', 'status.json', 'identity.json', 'writer.lock']
                                    and not p.name.endswith('.tmp')] +
                                   [record(root / name / 'summary.json') for name in functions]}
            write(root / 'summary.json', result)
            write(root / 'status.json', {**state, 'status': 'complete', 'finished_at': now()})
            return result
        except BaseException as error:
            write(root / 'status.json', {**state, 'status': 'failed', 'error': str(error), 'failed_at': now()})
            raise
