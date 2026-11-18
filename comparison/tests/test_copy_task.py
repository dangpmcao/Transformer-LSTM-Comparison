"""Copy(L): span sampling never crosses EOS, is deterministic, persists before scoring."""

import json

import numpy as np
import pytest

from comparison.copy_task import (
    NATURAL, RANDOM, build_manifest, load_manifest, load_rules, sample_spans, score_spans, span_gain, summarize,
    valid_starts,
)

EOS = 2
RULES = load_rules()


def test_protocol_rules_are_the_declared_ones():
    assert RULES["lengths"] == [25, 50, 100, 200] and RULES["primary_length"] == 100
    assert RULES["examples_per_length_per_source"] == 1000 and RULES["seed"] == 2718


def test_valid_starts_stay_inside_articles_and_before_eos():
    # Article 0: tokens 0..5 (EOS at 5); article 1: 6..8 (EOS at 8); article 2: 9..15 (EOS at 15).
    offsets = np.array([0, 6, 9, 16])
    assert valid_starts(offsets, 3).tolist() == [0, 1, 2, 9, 10, 11, 12]
    assert valid_starts(offsets, 6).tolist() == [9]
    assert valid_starts(offsets, 7).size == 0


def split_tokens(seed=0, articles=30):
    rng = np.random.default_rng(seed)
    tokens, offsets = [], [0]
    for _ in range(articles):
        length = int(rng.integers(5, 300))
        tokens += rng.integers(4, 16384, length).tolist() + [EOS]
        offsets.append(len(tokens))
    return np.array(tokens, dtype="<u2"), np.array(offsets, dtype="<u8")


@pytest.mark.parametrize("source", [NATURAL, RANDOM])
def test_spans_are_deterministic_and_never_contain_control_ids(source):
    tokens, offsets = split_tokens()
    first = sample_spans(source, 25, 200, 2718, tokens, offsets)
    again = sample_spans(source, 25, 200, 2718, tokens, offsets)
    assert all(np.array_equal(first[k], again[k]) for k in first)
    assert first["a"].shape == first["b"].shape == (200, 25)
    assert (first["a"] >= 4).all() and (first["b"] >= 4).all()
    assert not np.array_equal(first["a"], first["b"])  # A and B are independent draws.
    if source == NATURAL:
        for start, span in zip(first["a_start"], first["a"], strict=True):
            assert np.array_equal(tokens[start:start + 25], span)


def test_each_length_has_its_own_stream():
    tokens, offsets = split_tokens()
    short = sample_spans(NATURAL, 25, 50, 2718, tokens, offsets)
    long = sample_spans(NATURAL, 50, 50, 2718, tokens, offsets)
    assert not np.array_equal(short["a_start"], long["a_start"])


def test_gains_compare_first_and_second_span_and_can_skip_four_positions():
    length = 10
    first = np.full(length, 5.0)
    first[:4] = 9.0
    second = np.full(length, 2.0)
    second[:4] = 1.0
    nll = {"repeat": np.tile(np.concatenate([first, second]), (3, 1)),
           "control": np.tile(np.concatenate([first, first]), (3, 1))}
    assert span_gain(nll["repeat"]) == pytest.approx(first.mean() - second.mean())
    assert span_gain(nll["repeat"], 4) == pytest.approx(3.0)
    summary = summarize(nll)
    assert summary["all_positions"]["control_gain"] == pytest.approx(0.0)
    assert summary["exclude_first_four"]["copy_minus_control"] == pytest.approx(3.0)
    assert summary["examples"] == 3


def test_score_spans_feeds_repeat_and_control_inputs_without_bos_or_eos():
    seen = []

    def sequence_nll(targets):
        seen.append(list(targets))
        return np.ones(len(targets))

    spans = {"a": np.array([[5, 6]]), "b": np.array([[7, 8]])}
    nll = score_spans(sequence_nll, spans)
    assert seen == [[5, 6, 5, 6], [5, 6, 7, 8]]
    assert nll["repeat"].shape == (1, 4)


def test_manifest_round_trip_and_refuses_overwrite(tmp_path):
    tokens, offsets = split_tokens(articles=40)
    root = tmp_path / "tokens"
    root.mkdir()
    np.save(root / "test.tokens.npy", tokens)
    np.save(root / "test.offsets.npy", offsets)
    (root / "manifest.json").write_text(json.dumps({"tokenizer_sha256": "x"}))
    output = tmp_path / "manifest_test"
    manifest = build_manifest(output, token_root=root)
    assert len(manifest["files"]) == 8
    loaded, spans = load_manifest(output)
    assert spans[NATURAL, 100]["a"].shape == (1000, 100) and spans[RANDOM, 200]["b"].shape == (1000, 200)
    with pytest.raises(FileExistsError):
        build_manifest(output, token_root=root)
