"""5-gram baseline: word mapping, corpus lines, and window metrics with fake scorers."""

import math

import numpy as np
import pytest

from comparison.protocol import DEFAULT_PROTOCOL, load_protocol
from dataset.windows import WINDOW_TARGETS, make_index
from ngram.lm import LN10, article_line, id_to_word, iter_article_lines, window_scorer
from ngram.evaluate import score_split, summarize_windows

BOS, EOS, PAD, UNK = 1, 2, 0, 3
CONTEXT = load_protocol(DEFAULT_PROTOCOL)["evaluation"]["context"]


def test_ln10_converts_kenlm_base10():
    assert LN10 == pytest.approx(math.log(10))


def test_word_mapping_uses_kenlm_end_marker_for_eos():
    assert id_to_word(523) == "523"
    assert id_to_word(EOS) == "</s>"
    for control in (BOS, PAD, UNK):
        with pytest.raises(ValueError):
            id_to_word(control)


def test_article_line_drops_only_the_final_eos():
    assert article_line([10, 11, 12, EOS]) == "10 11 12"
    assert article_line([EOS]) == ""
    with pytest.raises(ValueError):
        article_line([10, 11])
    with pytest.raises(ValueError):
        article_line([10, EOS, 11, EOS])


def test_iter_article_lines_follows_offsets():
    tokens = np.array([5, 6, EOS, 7, EOS], dtype="<u2")
    offsets = np.array([0, 3, 5], dtype="<u8")
    assert list(iter_article_lines(tokens, offsets)) == ["5 6", "7"]


def fake_scorer(calls):
    """NLL depends only on the position inside the window: 1 + 1/(position+1)."""

    def target_log_probs(targets):
        calls.append(list(targets))
        positions = np.arange(len(targets))
        return -(1.0 + 1.0 / (positions + 1)), sum(token == 99 for token in targets)

    return target_log_probs


def test_window_scorer_sums_log_probabilities_per_window():
    score = window_scorer(fake_scorer([]))
    assert score([[4, 5], [4]]) == pytest.approx([-(2.0 + 1.5), -2.0])


def test_split_metrics_restart_each_window_and_exclude_short_windows():
    # Article 0 has 512 + 100 targets (two windows), article 1 has 512 targets (one full window).
    lengths = [WINDOW_TARGETS + 100, WINDOW_TARGETS]
    tokens, offsets = [], [0]
    for length in lengths:
        tokens += [10] * (length - 2) + [99, EOS]
        offsets.append(offsets[-1] + length)
    tokens, offsets = np.array(tokens, dtype="<u2"), np.array(offsets, dtype="<u8")
    index = make_index(offsets)
    calls = []
    scored = score_split(fake_scorer(calls), tokens, offsets, index)

    assert [len(call) for call in calls] == [512, 100, 512]  # Every window scored on its own.
    assert np.isnan(scored["nll"][1, 100:]).all() and not np.isnan(scored["nll"][1, :100]).any()
    summary = summarize_windows(scored["nll"], scored["oov"], index, CONTEXT)

    expected_total = 2 * sum(1 + 1 / (p + 1) for p in range(512)) + sum(1 + 1 / (p + 1) for p in range(100))
    assert summary["targets"] == 1124
    assert summary["mean_nll"] == pytest.approx(expected_total / 1124)
    assert summary["perplexity"] == pytest.approx(math.exp(expected_total / 1124))
    assert summary["oov_targets"] == 2  # One 99 at the end of each article; the short window holds one.
    context = summary["context"]
    assert (context["full_windows"], context["excluded_short_windows"]) == (2, 1)
    early = np.mean([1 + 1 / p for p in range(1, 33)])
    late = np.mean([1 + 1 / p for p in range(385, 513)])
    assert context["context_gain"] == pytest.approx(early - late)
    assert summary["position_mean_nll"][0] == pytest.approx(2.0)
