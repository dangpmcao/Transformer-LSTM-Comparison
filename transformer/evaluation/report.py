"""Seed-level reporting, scaling plots and optional matched LSTM comparisons."""
from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from .common import inside, read, write, sha, verify_record
from .runner import verify, result_valid
from .statistics import seed_summary, architecture_tests


def endpoints(result: dict) -> dict:
    m = result['metrics']
    return {'test_nll': m['test']['nll'], 'test_ppl': m['test']['perplexity'],
            'context_gain': m['test']['context_gain'],
            'natural_copy_100': m['copy']['natural_100']['gain'],
            'random_copy_100': m['copy']['random_100']['gain'],
            'blimp_accuracy': m['blimp']['primary']['overall']['accuracy'],
            **{f'same_target_{k}': v for k, v in m['same_target']['gains_against_full'].items()}}


def matched_quality(transformer: list[dict], lstm: list[dict], tolerance: float = .01) -> list[dict]:
    """Select using validation only; never interpolate unavailable checkpoint weights."""
    rows = []
    for size in ['s', 'm', 'l']:
        for seed in [13, 37, 73]:
            candidates = []
            for t in transformer:
                for r in lstm:
                    if t['size'] == r['size'] == size and t['seed'] == r['seed'] == seed:
                        relative = abs(math.expm1(t['validation_nll'] - r['validation_nll']))
                        if relative <= tolerance:
                            candidates.append((relative, t['view'], r['view'], t, r))
            if not candidates:
                rows.append({'size': size, 'seed': seed, 'status': 'no_retained_checkpoint_overlap',
                             'relative_ppl_tolerance': tolerance})
            else:
                relative, _, _, t, r = min(candidates, key=lambda c: c[:3])
                a, b = endpoints(t), endpoints(r)
                rows.append({'size': size, 'seed': seed, 'status': 'matched', 'relative_ppl_gap': relative,
                             'transformer_view': t['view'], 'lstm_view': r['view'],
                             'transformer_minus_lstm': {k: a[k] - b[k] for k in a}})
    return rows


def plots(root: Path, group: str, results: list[dict]) -> None:
    os.environ['MPLCONFIGDIR'] = str(inside(root / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for size in ['s', 'm', 'l']:
        selected = [r for r in results if r['size'] == size]
        if not selected:
            continue
        curves = np.asarray([r['metrics']['test']['position_mean_nll'] for r in selected])
        mean = curves.mean(axis=0)
        x = np.arange(1, len(mean) + 1)
        axes[0].plot(x, mean, label=f'{size.upper()} (n={len(selected)})')
        values = [endpoints(r)['blimp_accuracy'] * 100 for r in selected]
        axes[1].scatter([['s', 'm', 'l'].index(size)] * len(values), values)
        axes[1].plot([['s', 'm', 'l'].index(size)], [np.mean(values)], marker='_', markersize=18)
    axes[0].set(xlabel='Position in independent test window', ylabel='Mean NLL (nats/target)')
    axes[0].legend()
    axes[1].set(xticks=[0, 1, 2], xticklabels=['S', 'M', 'L'], ylabel='BLiMP accuracy (%)')
    fig.suptitle(group)
    fig.tight_layout()
    folder = inside(root / 'figures');folder.mkdir(exist_ok=True)
    fig.savefig(folder / f'{group}.png', dpi=180)
    fig.savefig(folder / f'{group}.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for axis, metric in zip(axes, ['test_ppl', 'context_gain', 'natural_copy_100'], strict=True):
        for i, size in enumerate(['s', 'm', 'l']):
            values = [endpoints(r)[metric] for r in results if r['size'] == size]
            if values:
                summary = seed_summary(values)
                axis.scatter([i] * len(values), values)
                axis.plot([i], [summary['mean']], marker='_', markersize=18)
        axis.set(xticks=[0, 1, 2], xticklabels=['S', 'M', 'L'], ylabel=metric)
    fig.suptitle(group + ' — individual seeds and means')
    fig.tight_layout()
    fig.savefig(folder / f'{group}_scaling.png', dpi=180)
    fig.savefig(folder / f'{group}_scaling.svg')
    plt.close(fig)


def collect(manifest: Path, comparison: Path | None = None, make_plots: bool = True) -> dict:
    manifest = inside(manifest)
    frozen = verify(manifest)
    results, missing = [], []
    for task in frozen['tasks']:
        identity = {'manifest_sha256': sha(manifest), 'task_index': task['index'],
                    'checkpoint_sha256': task['checkpoint']['sha256']}
        directory = manifest.parent / 'tasks' / task['directory']
        result = result_valid(directory, identity)
        if result:
            results.append(result)
        else:
            status = read(directory / 'status.json') if (directory / 'status.json').exists() else {'status': 'not_started'}
            missing.append({'index': task['index'], 'task': task['directory'], 'saved_status': status})
    aggregates = []
    for campaign in [g['name'] for g in frozen['settings']['campaigns']]:
        expected_seeds = next(g['seeds'] for g in frozen['settings']['campaigns'] if g['name'] == campaign)
        for view in frozen['settings']['checkpoint_views']:
            group = [r for r in results if r['campaign'] == campaign and r['view'] == view]
            for size in ['s', 'm', 'l']:
                selected = [r for r in group if r['size'] == size]
                if selected:
                    aggregates.append({'campaign': campaign, 'size': size, 'view': view,
                        'seeds': sorted(r['seed'] for r in selected),
                        'complete_seed_group': sorted(r['seed'] for r in selected) == sorted(expected_seeds),
                        'metrics': {k: seed_summary([endpoints(r)[k] for r in selected])
                                    for k in endpoints(selected[0])}})
            if make_plots and group:
                plots(manifest.parent, f'{campaign}_{view}', group)
    report = {'status': 'complete' if not missing else 'incomplete', 'completed_checkpoint_tasks': len(results),
              'expected_checkpoint_tasks': len(frozen['tasks']), 'missing': missing, 'aggregates': aggregates,
              'architecture_comparison': {'status': 'requires_external_lstm_results'},
              'baseline_controls': {'kenlm': 'requires_external_results', 'm_deep_lstm': 'requires_external_results'},
              'coverage': frozen['coverage'],
              'uncertainty': 'Sample SD and Student-t seed intervals describe independent training seeds; '
                             'item bootstrap intervals are conditional on each checkpoint. '
                             'Microbatch reruns and historical seed 101 are separate groups.',
              'proposal_comparison_status': 'awaiting_lstm_and_baseline_evidence',
              'compute': 'FLOPs are labelled analytic dense-matmul estimates; timing is from training logs.'}
    shared_hashes = {Path(r['path']).name: r['sha256'] for r in frozen['shared']}
    if comparison:
        external = read(comparison)
        if (external['architecture'] != 'lstm' or external['shared_manifest_hashes'] != shared_hashes
                or external['tokenizer_sha256'] != frozen['tokenizer_sha256']
                or external['evaluation_protocol_sha256'] != frozen['evaluation_protocol_sha256']):
            raise ValueError('External comparison used different targets/tokenizer/evaluation rules')
        for r in external['results']:
            if not r.get('checkpoint_sha256'):
                raise ValueError('External checkpoint provenance is missing')
        primary = [r for r in results if r['campaign'] == frozen['settings']['primary_campaign']]
        def paired(rows):
            side = {size: {} for size in ['s', 'm', 'l']}
            for r in rows:
                if r['view'] == 'best_validation' and r['size'] in side:
                    if r['seed'] in side[r['size']]:
                        raise ValueError('Duplicate independent seed in external comparison')
                    side[r['size']][r['seed']] = endpoints(r)
            return side
        tests = architecture_tests(paired(primary), paired(external['results']))
        report['architecture_comparison'] = {'status': 'computed', 'tests': tests,
            'external_bundle_sha256': sha(comparison), 'primary_campaign': frozen['settings']['primary_campaign'],
            'matched_validation_ppl': matched_quality(primary, external['results'])}
        cases = []
        pairs_manifest = read(manifest.parent / 'shared/blimp.json')['pairs']
        for r in external['results']:
            if r['view'] != 'best_validation' or r['size'] not in ['s', 'm', 'l']:
                continue
            local = next(x for x in primary if x['view'] == r['view'] and x['seed'] == r['seed'] and x['size'] == r['size'])
            task = next(t for t in frozen['tasks'] if t['campaign'] == local['campaign'] and t['seed'] == local['seed'] and t['size'] == local['size'] and t['view'] == local['view'])
            if 'pairs_file' not in r:
                continue
            verify_record(r['pairs_file'])
            left = [json.loads(l) for l in (manifest.parent / 'tasks' / task['directory'] / 'blimp/pairs.jsonl').read_text().splitlines()]
            right = [json.loads(l) for l in Path(r['pairs_file']['path']).read_text().splitlines()]
            if [(p['paradigm'], p['pair_id']) for p in left] != [(p['paradigm'], p['pair_id']) for p in right]:
                raise ValueError('BLiMP comparison pair order differs')
            for pair, t, l in zip(pairs_manifest, left, right, strict=True):
                if t['correct'] != l['correct']:
                    cases.append({'size': r['size'], 'seed': r['seed'], **pair,
                                  'transformer_correct': t['correct'], 'lstm_correct': l['correct']})
                    if len(cases) >= frozen['settings']['case_examples']:
                        break
        report['architecture_comparison']['blimp_disagreement_cases'] = cases
        report['proposal_comparison_status'] = 'architecture_tests_computed_controls_still_require_verification'
        parameter_checks = []
        for ext in external['results']:
            if ext['view'] != 'best_validation' or ext['size'] not in ['s', 'm', 'l']:
                continue
            local = next(r for r in primary if r['size'] == ext['size'] and r['seed'] == ext['seed'] and r['view'] == ext['view'])
            count = ext.get('parameters', {}).get('non_embedding')
            reference = local['resources']['parameters']['non_embedding']
            parameter_checks.append({'size': ext['size'], 'seed': ext['seed'],
                'relative_parameter_gap': abs(count-reference)/reference if count is not None else None,
                'within_five_percent': abs(count-reference)/reference <= .05 if count is not None else None})
        report['architecture_comparison']['parameter_matching'] = parameter_checks
        if 'kenlm' in external.get('baselines', {}):
            baseline = external['baselines']['kenlm']
            if not baseline.get('model_sha256') or not baseline.get('metrics'):
                raise ValueError('KenLM baseline needs actual model hash and measurements')
            report['baseline_controls']['kenlm'] = baseline
        report['architecture_comparison']['case_status'] = 'computed' if cases else 'no_disagreement_or_pair_outputs_missing'
        if any(r['size'] == 'm_deep' for r in external['results']):
            report['baseline_controls']['m_deep_lstm'] = [
                {k: r[k] for k in ['size', 'seed', 'view', 'validation_nll', 'metrics']}
                for r in external['results'] if r['size'] == 'm_deep']
    # Export one template contract and machine-readable per-run endpoints for the LSTM owner.
    write(manifest.parent / 'comparison_contract.json', {
        'schema_version': 1, 'shared_manifest_hashes': shared_hashes,
        'evaluation_protocol_sha256': frozen['evaluation_protocol_sha256'],
        'tokenizer_sha256': frozen['tokenizer_sha256'],
        'required_lstm_seeds': [13, 37, 73], 'sizes': ['s', 'm', 'l'],
        'views': frozen['settings']['checkpoint_views'],
        'result_fields': ['size', 'seed', 'view', 'validation_nll', 'checkpoint_sha256', 'metrics'],
        'optional_fields': ['pairs_file (path, bytes, sha256)', 'm_deep results'],
        'metrics_schema': 'Same metrics keys as each tasks/*/*/*/*/summary.json; use the frozen shared targets.'})
    write(manifest.parent / 'report.json', report)
    write(manifest.parent / 'run_results.json', {'results': results})
    path = inside(manifest.parent / 'results.csv')
    with path.open('w', newline='') as stream:
        fields = ['campaign', 'size', 'seed', 'view', 'validation_nll', *endpoints(results[0])] if results else ['campaign', 'size', 'seed', 'view']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields if k in r} | endpoints(r))
    return report
