"""Same-target histories: exact truncation, local-order-preserving shuffle, determinism."""

import numpy as np
import pytest

from comparison.same_target import (
    FULL, SHUFFLE, distant_permutation, history, iter_target_histories, load_rules, summarize_scores,
)

RULES = load_rules()
EOS = 2


def window_tokens():
    return np.arange(10, 10 + 512, dtype=np.int64)  # Distinct IDs make positions recoverable.


def test_protocol_rules_are_the_declared_ones():
    assert (RULES.first_position, RULES.last_position) == (385, 512)
    assert RULES.limits == (4, 32, 128) and RULES.local_tokens == 128 and RULES.seed == 1618
    assert RULES.conditions == ("full", "k4", "k32", "k128", "distant_shuffle_128")


@pytest.mark.parametrize("position", [385, 450, 512])
def test_full_and_truncated_histories_never_include_target_or_future(position):
    window = window_tokens()
    assert np.array_equal(history(window, position, FULL, RULES), window[:position - 1])
    for limit in RULES.limits:
        prefix = history(window, position, f"k{limit}", RULES)
        assert np.array_equal(prefix, window[position - 1 - limit:position - 1])  # Nearest k, original order.
    assert window[position - 1] not in history(window, position, FULL, RULES)


def test_shuffle_keeps_length_local_suffix_and_permutes_only_distant_tokens():
    window, position = window_tokens(), 450
    permutation = distant_permutation(RULES, article=3, window_start=512, position=position)
    shuffled = history(window, position, SHUFFLE, RULES, permutation)
    prefix = window[:position - 1]
    distant = len(prefix) - 128
    assert len(shuffled) == len(prefix)
    assert np.array_equal(shuffled[distant:], prefix[distant:])
    assert sorted(shuffled[:distant]) == sorted(prefix[:distant])
    assert not np.array_equal(shuffled[:distant], prefix[:distant])


def test_permutations_are_seeded_by_target_identity():
    a = distant_permutation(RULES, 3, 512, 450)
    assert np.array_equal(a, distant_permutation(RULES, 3, 512, 450))
    assert not np.array_equal(a, distant_permutation(RULES, 4, 512, 450))
    assert len(distant_permutation(RULES, 0, 0, 385)) == 385 - 1 - 128


def test_iter_histories_checks_tokens_against_manifest():
    tokens = np.concatenate([window_tokens(), [EOS]]).astype("<u2")
    offsets = np.array([0, len(tokens)], dtype="<u8")
    targets = np.array([[0, 0, 400, 10 + 399]], dtype="<i8")
    row, target, histories = next(iter_target_histories(targets, tokens, offsets, RULES))
    assert (row, target) == (0, 409) and set(histories) == set(RULES.conditions)
    with pytest.raises(ValueError):
        next(iter_target_histories(np.array([[0, 0, 400, 999]]), tokens, offsets, RULES))


def test_summary_gains_are_paired_differences():
    nll = {c: np.array([1.0, 2.0]) for c in RULES.conditions}
    nll["k4"] = np.array([1.5, 2.5])
    summary = summarize_scores(nll, RULES.conditions)
    assert summary["truncation_gain"]["k4"] == pytest.approx(0.5)
    assert summary["truncation_gain"]["k128"] == 0 and summary["shuffle_gain"] == 0
    nll["k32"] = np.array([1.0, np.nan])
    with pytest.raises(ValueError):
        summarize_scores(nll, RULES.conditions)
