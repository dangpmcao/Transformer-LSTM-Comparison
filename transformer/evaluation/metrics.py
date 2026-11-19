"""Proposal measurements on shared frozen targets; no fitting or model selection."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from comparison.blimp_scorer import summarize
from .common import inside, read, save_arrays, write
from .manifests import shuffled_history
from .statistics import accuracy_ci, cluster_ci


def test_metrics(scorer, shared: Path, output: Path, settings: dict) -> dict:
    with np.load(shared / 'test.npz', allow_pickle=False) as data:
        targets, index = data['targets'], data['index']
    losses = scorer.windows([x[:int(row[2])].tolist() for x, row in zip(targets, index, strict=True)])
    matrix = np.zeros(targets.shape, dtype=np.float64)
    for i, loss in enumerate(losses):
        matrix[i, :len(loss)] = loss
    save_arrays(output / 'scores.npz', nll=matrix, index=index)
    return summarize_test(matrix, index, settings)


def summarize_test(nll: np.ndarray, index: np.ndarray, settings: dict) -> dict:
    full = index[:, 2] == nll.shape[1]
    if not full.any():
        raise ValueError('No full test windows for the position curve')
    counts = index[:, 2]
    valid = np.arange(nll.shape[1])[None, :] < counts[:, None]
    if not np.isfinite(nll[valid]).all():
        raise FloatingPointError('Invalid test scores')
    total = float(np.where(valid, nll, 0).sum())
    mean = total / int(counts.sum())
    curve = nll[full].mean(axis=0)
    gain = nll[full, :32].mean(axis=1) - nll[full, 384:512].mean(axis=1)
    return {'nll': mean, 'perplexity': math.exp(mean), 'targets': int(counts.sum()),
            'full_windows': int(full.sum()), 'short_windows': int((~full).sum()),
            'position_mean_nll': curve.tolist(), 'context_gain': float(gain.mean()),
            'article_bootstrap_ppl_ci95': cluster_ci(np.where(valid, nll, 0).sum(axis=1),
                counts, index[:, 0], settings['bootstrap_replicates'], settings['bootstrap_seed'], np.exp),
            'article_bootstrap_context_ci95': cluster_ci(gain, np.ones(len(gain)),
                index[full, 0], settings['bootstrap_replicates'], settings['bootstrap_seed'])}


def blimp_metrics(scorer, shared: Path, output: Path, settings: dict) -> dict:
    manifest = read(shared / 'blimp.json')
    losses = scorer.windows(manifest['windows'])
    scores = [-float(x.sum()) for x in losses]
    results = []
    for pair in manifest['pairs']:
        good = sum(scores[i] for i in pair['good_windows'])
        bad = sum(scores[i] for i in pair['bad_windows'])
        results.append({key: pair[key] for key in ['paradigm', 'pair_id', 'group',
                        'good_targets', 'bad_targets', 'identical_candidates']} |
                       {'good_log_probability': good, 'bad_log_probability': bad,
                        'correct': good > bad,
                        'length_normalized_correct': good / pair['good_targets'] > bad / pair['bad_targets']})
    temporary = inside(output / 'pairs.jsonl.tmp')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('w') as stream:
        import json
        for result in results:
            stream.write(json.dumps(result, allow_nan=False) + '\n')
    temporary.replace(output / 'pairs.jsonl')
    result = summarize(results)
    for view in [result['primary'], *result['sensitivities'].values()]:
        for row in [view['overall'], *view['groups'].values(), *view['paradigms'].values()]:
            row['pair_bootstrap_ci95'] = accuracy_ci(row['correct'], row['pairs'],
                settings['bootstrap_replicates'], settings['bootstrap_seed'])
    margins = np.asarray([r['good_log_probability'] - r['bad_log_probability'] for r in results])
    selected = np.argsort(margins, kind='stable')[:settings['case_examples']]
    result['most_confident_errors_or_smallest_margins'] = [
        {'paradigm': manifest['pairs'][i]['paradigm'], 'pair_id': manifest['pairs'][i]['pair_id'],
         'sentence_good': manifest['pairs'][i]['sentence_good'],
         'sentence_bad': manifest['pairs'][i]['sentence_bad'], 'margin': float(margins[i])} for i in selected]
    return result


def copy_metrics(scorer, shared: Path, output: Path, settings: dict) -> dict:
    result = {}
    for source in ['natural', 'random']:
        for length in settings['copy_lengths']:
            with np.load(shared / f'copy_{source}_{length}.npz', allow_pickle=False) as data:
                first, independent, article = data['first'], data['independent'], data['article']
            repeated = np.concatenate([first, first], axis=1)
            control = np.concatenate([first, independent], axis=1)
            repeated_loss = np.stack(scorer.windows(repeated.tolist()))
            control_loss = np.stack(scorer.windows(control.tolist()))
            save_arrays(output / f'{source}_{length}.npz', repeated_nll=repeated_loss,
                        independent_nll=control_loss, article=article)
            row = copy_summary(repeated_loss, control_loss, length)
            # Natural spans are clustered by sampled source article. Random sequences are separate units.
            clusters = article if source == 'natural' else np.arange(len(article))
            gain = repeated_loss[:, :length].mean(axis=1) - repeated_loss[:, length:].mean(axis=1)
            row['source_article_or_random_example_bootstrap_ci95'] = cluster_ci(
                gain, np.ones(len(gain)), clusters,
                settings['bootstrap_replicates'], settings['bootstrap_seed'])
            row['first_span_nll'] = float(repeated_loss[:, :length].mean())
            row['repeated_span_nll'] = float(repeated_loss[:, length:].mean())
            result[f'{source}_{length}'] = row
    return result


def copy_summary(repeated: np.ndarray, independent: np.ndarray, length: int) -> dict:
    if repeated.shape != independent.shape or repeated.ndim != 2 or repeated.shape[1] != 2 * length:
        raise ValueError('Expected paired copy scores [examples, 2*length]')
    if length <= 4 or not np.isfinite(repeated).all() or not np.isfinite(independent).all():
        raise ValueError('Invalid copy scores')
    gain = repeated[:, :length].mean(axis=1) - repeated[:, length:].mean(axis=1)
    control = independent[:, :length].mean(axis=1) - independent[:, length:].mean(axis=1)
    adjusted = repeated[:, 4:length].mean(axis=1) - repeated[:, length + 4:].mean(axis=1)
    adjusted_control = independent[:, 4:length].mean(axis=1) - independent[:, length + 4:].mean(axis=1)
    return {'examples': len(gain), 'gain': float(gain.mean()),
            'independent_span_gain': float(control.mean()),
            'gain_minus_independent': float((gain - control).mean()),
            'exclude_first_four_gain': float(adjusted.mean()),
            'exclude_first_four_independent_gain': float(adjusted_control.mean()),
            'exclude_first_four_gain_minus_independent': float((adjusted - adjusted_control).mean())}


def same_target_metrics(scorer, shared: Path, output: Path, settings: dict, full_scores: Path) -> dict:
    with np.load(shared / 'test.npz', allow_pickle=False) as data:
        targets, index = data['targets'], data['index']
        windows, positions = data['same_window'], data['same_position']
    with np.load(full_scores, allow_pickle=False) as data:
        baseline = data['nll'][windows, positions]
    conditions = ['full', 'k4', 'k32', 'k128', 'shuffle128']
    result = np.empty((len(windows), len(conditions)), dtype=np.float64)
    result[:, 0] = baseline
    unchanged = 0
    for start in range(0, len(windows), settings['target_chunk_size']):
        end = min(start + settings['target_chunk_size'], len(windows))
        path = output / 'chunks' / f'{start:07d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                chunk, count = data['scores'], int(data['unchanged'])
                if not np.array_equal(data['windows'], windows[start:end]) or not np.array_equal(data['positions'], positions[start:end]):
                    raise ValueError('Resume chunk target order changed')
            if chunk.shape != (end - start, 4) or not np.isfinite(chunk).all():
                raise ValueError('Invalid saved same-target chunk')
        else:
            truth = [int(targets[w, p]) for w, p in zip(windows[start:end], positions[start:end], strict=True)]
            columns = []
            for limit in [4, 32, 128]:
                inputs = [[1, *targets[w, p - limit:p].tolist()]
                          for w, p in zip(windows[start:end], positions[start:end], strict=True)]
                columns.append(scorer.last(inputs, truth))
            inputs = [shuffled_history(targets[w], int(p), int(index[w, 0]), int(index[w, 1]),
                                      settings['shuffle_seed'])
                      for w, p in zip(windows[start:end], positions[start:end], strict=True)]
            count = sum(np.array_equal(x[1:], targets[w, :p])
                        for x, w, p in zip(inputs, windows[start:end], positions[start:end], strict=True))
            columns.append(scorer.last(inputs, truth))
            chunk = np.stack(columns, axis=1)
            save_arrays(path, scores=chunk, windows=windows[start:end], positions=positions[start:end],
                        unchanged=np.asarray(count))
        unchanged += count
        result[start:end, 1:] = chunk
        print(f'same_target {end}/{len(windows)}', flush=True)
    save_arrays(output / 'scores.npz', nll=result, window=windows, position=positions,
                article=index[windows, 0], target_id=targets[windows, positions])
    gains = result[:, 1:] - result[:, :1]
    return {'conditions': conditions, 'targets': len(windows),
            'condition_mean_nll': dict(zip(conditions, result.mean(axis=0).tolist(), strict=True)),
            'gains_against_full': dict(zip(conditions[1:], gains.mean(axis=0).tolist(), strict=True)),
            'article_bootstrap_gain_ci95': {condition: cluster_ci(gains[:, i], np.ones(len(windows)),
                index[windows, 0], settings['bootstrap_replicates'], settings['bootstrap_seed'])
                for i, condition in enumerate(conditions[1:])},
            'unchanged_shuffle_targets': unchanged}
