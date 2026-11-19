"""Analytic checks, boundary controls, paired inference and resumable CPU evaluation."""
from copy import deepcopy
from pathlib import Path
import json
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from transformer.model import TransformerConfig, TransformerLM
from transformer.evaluation.common import save_arrays, write, read, inside
from transformer.evaluation.scoring import Scorer
from transformer.evaluation.manifests import copy_sample, shuffled_history
from transformer.evaluation.metrics import summarize_test, copy_summary
from transformer.evaluation.statistics import seed_summary, holm, architecture_tests
from transformer.evaluation.report import matched_quality
from transformer.evaluation import runner


def tiny():
    torch.manual_seed(13)
    return TransformerLM(TransformerConfig(vocab_size=16, max_positions=512,
                         layers=1, width=8, heads=2, dropout=0.1)).eval()


@pytest.mark.parametrize('batch', [1, 3])
def test_final_logits_match_full_logits_with_right_padding(batch):
    model = tiny()
    inputs = [[1, 4, 5, 6], [1, 7], [1, 8, 9]]
    targets = [7, 8, 10]
    scorer = Scorer(model, 'cpu', batch)
    actual = scorer.last(inputs, targets)
    expected = []
    for ids, target in zip(inputs, targets, strict=True):
        logits = model(torch.tensor([ids]))[:, -1, :]
        expected.append(float(F.cross_entropy(logits, torch.tensor([target]))))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    scores = scorer.windows([[4, 5, 6, 7], [7, 8], [8, 9, 10]])
    np.testing.assert_allclose([x[-1] for x in scores], actual, rtol=1e-6, atol=1e-6)
    assert all(len(x) == n for x, n in zip(scores, [4, 2, 3], strict=True))


def test_shuffle_preserves_positions_local_suffix_and_target_exclusion():
    targets = np.arange(512) + 4
    a = shuffled_history(targets, 400, 3, 512)
    b = shuffled_history(targets, 400, 3, 512)
    np.testing.assert_array_equal(a, b)
    assert a[0] == 1 and len(a) == 401
    np.testing.assert_array_equal(a[-128:], targets[272:400])
    np.testing.assert_array_equal(np.sort(a[1:273]), targets[:272])
    assert targets[400] not in a and not np.array_equal(a[1:273], targets[:272])


def test_natural_copy_never_samples_eos_or_crosses_article():
    tokens = np.array([4] * 30 + [2] + [7] * 40 + [2])
    offsets = np.array([0, 31, 72], dtype=np.uint64)
    samples, article, local = copy_sample(tokens, offsets, 25, 300, np.random.default_rng(2718))
    assert samples.shape == (300, 25) and set(article) == {0, 1}
    for span, a, start in zip(samples, article, local, strict=True):
        assert np.all(span == [4, 7][a]) and start + 25 <= [30, 40][a]


def test_test_metric_denominators_include_short_windows_without_padding():
    nll = np.full((2, 512), 2.5)
    nll[1, 10:] = 999
    index = np.array([[0, 0, 512], [1, 0, 10]])
    result = summarize_test(nll, index, {'bootstrap_replicates': 100, 'bootstrap_seed': 31415})
    assert result['targets'] == 522 and result['full_windows'] == 1
    assert result['nll'] == 2.5 and result['context_gain'] == 0
    assert result['perplexity'] == pytest.approx(math.exp(2.5))
    assert result['position_mean_nll'] == [2.5] * 512


def test_copy_gain_sign_and_independent_and_first_four_controls():
    repeated = np.c_[np.full((3, 25), 2.), np.ones((3, 25))]
    control = np.full((3, 50), 2.)
    result = copy_summary(repeated, control, 25)
    assert result['gain'] == result['gain_minus_independent'] == result['exclude_first_four_gain'] == 1
    assert result['independent_span_gain'] == 0


def test_seed_uncertainty_and_holm_complete_families():
    assert seed_summary([1])['seed_t_ci95'] is None
    summary = seed_summary([1, 2, 3])
    assert summary['sample_sd'] == 1 and summary['mean'] == 2
    assert summary['seed_t_ci95'][0] < 0
    assert holm([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    with pytest.raises(ValueError):
        holm([float('nan')])
    a, b = {}, {}
    for index, size in enumerate(['s', 'm', 'l'], 1):
        a[size], b[size] = {}, {}
        for j, seed in enumerate([13, 37, 73], 1):
            b[size][seed] = {'context_gain': 1., 'natural_copy_100': 1., 'blimp_accuracy': .6}
            a[size][seed] = {'context_gain': 1 + index * (1 + .1 * j),
                             'natural_copy_100': 1 + index * (1 + .15 * j),
                             'blimp_accuracy': .6 + .0001 * j}
    result = architecture_tests(a, b)
    assert len(result['h1']) == 6 and len(result['h2_tost_percentage_points']) == 3 and len(result['h3']) == 2
    assert all(x['reject_at_005'] for x in result['h2_tost_percentage_points'])
    del b['s'][13]
    with pytest.raises(ValueError):
        architecture_tests(a, b)


def test_quality_matching_does_not_fabricate_missing_weights():
    result = matched_quality([], [])
    assert len(result) == 9 and all(x['status'] == 'no_retained_checkpoint_overlap' for x in result)
    with pytest.raises(ValueError, match='inside'):
        inside(Path('/tmp/forbidden-evaluation-output'))


def test_real_cpu_runner_all_metrics_resume_and_artifact_integrity(tmp_path, monkeypatch):
    shared = tmp_path / 'shared'
    targets = np.tile(np.array([4, 5, 6, 7], dtype=np.int64), (1, 128))
    save_arrays(shared / 'test.npz', targets=targets, index=np.array([[0, 0, 512]]),
                same_window=np.array([0, 0]), same_position=np.array([384, 511]))
    rng = np.random.default_rng(2718)
    for source in ['natural', 'random']:
        for length in [25, 50, 100, 200]:
            save_arrays(shared / f'copy_{source}_{length}.npz',
                        first=rng.integers(4, 16, (2, length)), independent=rng.integers(4, 16, (2, length)),
                        article=np.array([0, 1]))
    write(shared / 'blimp.json', {'windows': [[4, 5, 2], [4, 6, 2]], 'pairs': [
        {'paradigm': 'toy', 'pair_id': '0', 'group': 'toy', 'sentence_good': 'good', 'sentence_bad': 'bad',
         'good_windows': [0], 'bad_windows': [1], 'good_targets': 3, 'bad_targets': 3,
         'identical_candidates': False}]})
    task = {'index': 0, 'directory': 'toy/s/seed13/best_validation', 'campaign': 'toy',
            'size': 's', 'seed': 13, 'view': 'best_validation', 'validation_nll': 1.,
            'checkpoint': {'path': 'fixture', 'sha256': 'fixture'}, 'resources': {}}
    settings = {'inference_batch_size': 2, 'target_chunk_size': 1, 'copy_lengths': [25, 50, 100, 200],
                'copy_examples': 2, 'shuffle_seed': 1618, 'bootstrap_replicates': 20,
                'bootstrap_seed': 31415, 'case_examples': 1}
    frozen = {'tasks': [task], 'settings': settings}
    manifest = tmp_path / 'manifest.json';write(manifest, frozen)
    monkeypatch.setattr(runner, 'verify', lambda *args, **kwargs: frozen)
    monkeypatch.setattr(runner, 'load_model', lambda *args: tiny())
    result = runner.run(manifest, 0, 'cpu')
    assert result['status'] == 'complete'
    assert result['metrics']['test']['targets'] == 512
    assert result['metrics']['same_target']['targets'] == 2
    assert result['metrics']['blimp']['primary']['overall']['pairs'] == 1
    assert len(result['metrics']['copy']) == 8
    monkeypatch.setattr(runner, 'load_model', lambda *args: pytest.fail('Completed task must not reload model'))
    assert runner.run(manifest, 0, 'cpu') == result
    pair_file = tmp_path / 'tasks' / task['directory'] / 'blimp/pairs.jsonl'
    pair_file.write_text('corrupted')
    with pytest.raises(ValueError, match='changed'):
        runner.run(manifest, 0, 'cpu')


def test_checkpoint_rollbacks_do_not_count_discarded_updates_twice():
    events = [
        {'event': 'update', 'updates': 1, 'prediction_tokens': 5},
        {'event': 'validation', 'updates': 1, 'nll': 2.},
        {'event': 'update', 'updates': 2, 'prediction_tokens': 9},
        {'event': 'validation', 'updates': 2, 'nll': 1.},
        {'event': 'update', 'updates': 2, 'prediction_tokens': 7},
        {'event': 'validation', 'updates': 2, 'nll': 1.5},
        {'event': 'update', 'updates': 3, 'prediction_tokens': 8}]
    validation, updates, discarded = runner.canonical_history(events, 3)
    assert discarded == 1 and sum(e['prediction_tokens'] for e in updates) == 20
    assert validation[-1]['nll'] == 1.5
