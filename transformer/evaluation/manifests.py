"""Shared test targets, BPE pairs and copy samples, persisted before scoring."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from comparison.blimp_scorer import load_pairs, split_target_windows
from dataset.encode import split_arrays
from dataset.windows import load_windows
from tokenizer.core import encode_text, load_tokenizer
from .common import save_arrays, write


def copy_sample(tokens: np.ndarray, offsets: np.ndarray, length: int, count: int,
                rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniform over valid content starts, never across article boundaries/EOS."""
    valid = np.maximum(np.diff(offsets).astype(np.int64) - 1 - length + 1, 0)
    cumulative = np.cumsum(valid)
    if not len(cumulative) or cumulative[-1] <= 0:
        raise ValueError('No article long enough for the copy span')
    flat = rng.integers(0, int(cumulative[-1]), size=count)
    article = np.searchsorted(cumulative, flat, side='right')
    previous = np.r_[0, cumulative[:-1]]
    local = flat - previous[article]
    starts = offsets[article].astype(np.int64) + local
    samples = np.asarray([tokens[s:s + length] for s in starts], dtype=np.int64)
    if np.any(samples < 4):
        raise ValueError('Copy spans must contain content IDs only')
    return samples, article.astype(np.int64), local.astype(np.int64)


def shuffled_history(targets: np.ndarray, position: int, article: int, window_start: int,
                     seed: int = 1618, local: int = 128) -> np.ndarray:
    """position is zero-based: x_t's history excludes x_t and future tokens."""
    history = np.array(targets[:position], dtype=np.int64)
    distant = position - local
    if distant < 0:
        raise ValueError('Target has insufficient history')
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        [seed, int(article), int(window_start), position + 1])))
    history[:distant] = history[:distant][rng.permutation(distant)]
    return np.r_[1, history]


def build(root: Path, token_dir: Path, tokenizer_dir: Path, data_root: Path,
          settings: dict) -> dict:
    manifest = load_windows(token_dir / 'windows', token_dir)
    tokens, offsets = split_arrays(token_dir, 'test')
    index = np.load(token_dir / 'windows/test.windows.npy', allow_pickle=False)
    ids = np.zeros((len(index), 512), dtype=np.int64)
    for i, (article, local, valid) in enumerate(index):
        start = int(offsets[article]) + int(local)
        ids[i, :valid] = tokens[start:start + valid]
    full = np.flatnonzero(index[:, 2] == 512)
    windows = np.repeat(full, 128)
    positions = np.tile(np.arange(384, 512, dtype=np.int64), len(full))
    save_arrays(root / 'test.npz', targets=ids, index=index,
                same_window=windows, same_position=positions)
    rng = np.random.Generator(np.random.PCG64(settings['copy_seed']))
    count = settings['copy_examples']
    for source in ['natural', 'random']:
        for length in settings['copy_lengths']:
            if source == 'natural':
                first, article, local = copy_sample(tokens, offsets, length, count, rng)
                other, other_article, other_local = copy_sample(tokens, offsets, length, count, rng)
            else:
                first = rng.integers(4, 16384, size=(count, length), dtype=np.int64)
                other = rng.integers(4, 16384, size=(count, length), dtype=np.int64)
                article = other_article = np.full(count, -1, dtype=np.int64)
                local = other_local = np.full(count, -1, dtype=np.int64)
            save_arrays(root / f'copy_{source}_{length}.npz', first=first, independent=other,
                        article=article, local=local, independent_article=other_article,
                        independent_local=other_local)
    pairs = load_pairs(data_root)
    tokenizer = load_tokenizer(tokenizer_dir)
    unique: dict[tuple[int, ...], int] = {}
    encoded_pairs = []
    for pair in pairs:
        row = dict(pair)
        for label in ['good', 'bad']:
            targets = encode_text(tokenizer, row[f'sentence_{label}'], add_eos=True)
            window_ids = []
            for window in split_target_windows(targets):
                key = tuple(window)
                if key not in unique:
                    unique[key] = len(unique)
                window_ids.append(unique[key])
            row[f'{label}_windows'] = window_ids
            row[f'{label}_targets'] = len(targets)
        row['identical_candidates'] = row['good_windows'] == row['bad_windows']
        encoded_pairs.append(row)
    write(root / 'blimp.json', {'windows': [list(k) for k in unique], 'pairs': encoded_pairs})
    return {'test_windows': len(index), 'test_articles': len(offsets) - 1,
            'test_targets': int(index[:, 2].sum()), 'full_windows': len(full),
            'same_targets': len(windows), 'excluded_short_windows': len(index) - len(full),
            'excluded_short_window_targets': int(index[index[:, 2] < 512, 2].sum()),
            'blimp_pairs': len(pairs), 'blimp_unique_windows': len(unique),
            'blimp_groups': len({r['group'] for r in pairs}),
            'blimp_paradigms': len({r['paradigm'] for r in pairs}),
            'copy_examples_per_length_source': count,
            'window_manifest': manifest['splits']['test']}
