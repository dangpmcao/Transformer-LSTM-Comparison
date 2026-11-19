"""Seed uncertainty, item bootstrap, and prespecified architecture contrasts."""
from __future__ import annotations

import numpy as np
from scipy import stats


def seed_summary(values: list[float]) -> dict:
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Require finite seed measurements')
    mean = float(x.mean())
    if len(x) < 2:
        return {'n': len(x), 'mean': mean, 'sample_sd': None, 'seed_t_ci95': None}
    sd = float(x.std(ddof=1))
    radius = float(stats.t.ppf(.975, len(x) - 1) * sd / np.sqrt(len(x)))
    return {'n': len(x), 'mean': mean, 'sample_sd': sd,
            'seed_t_ci95': [mean - radius, mean + radius]}


def cluster_ci(numerators: np.ndarray, denominators: np.ndarray, clusters: np.ndarray,
               replicates: int = 10000, seed: int = 31415, transform=None) -> list[float]:
    """Resample independent article clusters; recompute the token-weighted ratio."""
    numerators, denominators = np.asarray(numerators), np.asarray(denominators)
    if (numerators.shape != denominators.shape or numerators.shape != clusters.shape
            or not np.isfinite(numerators).all() or not np.isfinite(denominators).all()
            or np.any(denominators <= 0) or replicates < 1):
        raise ValueError('Invalid bootstrap units')
    _, group = np.unique(clusters, return_inverse=True)
    a = np.bincount(group, weights=numerators)
    b = np.bincount(group, weights=denominators)
    rng = np.random.Generator(np.random.PCG64(seed))
    weights = rng.multinomial(len(a), np.ones(len(a)) / len(a), size=replicates)
    result = (weights @ a) / (weights @ b)
    if transform:
        result = transform(result)
    return np.quantile(result, [.025, .975]).tolist()


def accuracy_ci(correct: int, count: int, replicates: int = 10000, seed: int = 31415):
    """Exact distribution of the nonparametric pair bootstrap for a binary score."""
    if count == 0:
        return None
    if not 0 <= correct <= count:
        raise ValueError('Invalid accuracy count')
    rng = np.random.Generator(np.random.PCG64(seed))
    return np.quantile(rng.binomial(count, correct / count, size=replicates) / count,
                       [.025, .975]).tolist()


def holm(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    if not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Holm requires a complete finite prespecified family')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def paired_test(differences: list[float], margin: float | None = None) -> dict:
    x = np.asarray(differences, dtype=float)
    result = seed_summary(differences)
    if len(x) < 2 or result['sample_sd'] == 0:
        return {**result, 'status': 'insufficient_seed_variance', 'p': None}
    sem = float(stats.sem(x))
    if margin is None:
        p = float(stats.t.sf(x.mean() / sem, len(x) - 1))
    else:
        # TOST: both bounds must be rejected. Accuracy differences use percentage points.
        lower = stats.t.sf((x.mean() + margin) / sem, len(x) - 1)
        upper = stats.t.cdf((x.mean() - margin) / sem, len(x) - 1)
        p = float(max(lower, upper))
        radius = float(stats.t.ppf(.95, len(x) - 1) * sem)
        result.update(equivalence_margin=margin, tost_ci90=[float(x.mean()) - radius,
                                                         float(x.mean()) + radius])
    return {**result, 'status': 'computed', 'p': p}


def architecture_tests(transformer: dict, lstm: dict, seeds=(13, 37, 73)) -> dict:
    """Inputs: size -> seed -> {context_gain, natural_copy_100, blimp_accuracy}.

    Each architecture supplies one chosen campaign, never both microbatch reruns.
    Independent units are paired training seeds, not evaluation items.
    """
    for side in [transformer, lstm]:
        if any(size not in side or set(side[size]) != set(seeds) for size in ['s', 'm', 'l']):
            raise ValueError('Require all three paired seeds at every size')
    delta = {size: {metric: [transformer[size][seed][metric] - lstm[size][seed][metric]
                            for seed in seeds]
                    for metric in ['context_gain', 'natural_copy_100', 'blimp_accuracy']}
             for size in ['s', 'm', 'l']}
    h1 = [{'size': size, 'endpoint': metric, **paired_test(delta[size][metric])}
          for size in ['s', 'm', 'l'] for metric in ['context_gain', 'natural_copy_100']]
    h2 = [{'size': size, **paired_test([100 * d for d in delta[size]['blimp_accuracy']], 2.0)}
          for size in ['s', 'm', 'l']]
    h3 = [{'endpoint': metric, **paired_test(list(np.asarray(delta['l'][metric]) -
                                               np.asarray(delta['s'][metric])))}
          for metric in ['context_gain', 'natural_copy_100']]
    for family in [h1, h2, h3]:
        if all(row['p'] is not None for row in family):
            for row, adjusted in zip(family, holm([row['p'] for row in family]), strict=True):
                row.update(holm_p=adjusted, reject_at_005=adjusted < .05)
        else:
            for row in family:
                row.update(holm_p=None, reject_at_005=None)
    return {'h1': h1, 'h2_tost_percentage_points': h2, 'h3': h3}
