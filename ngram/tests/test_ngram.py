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


def test_windows_start_from_null_context_not_article_start(tmp_path):
    """Real KenLM: every training line starts with 7, so <s> would wrongly favour 7 mid-article."""
    kenlm = pytest.importorskip("kenlm")
    from ngram.lm import DEFAULT_KENLM_BIN, KenLMScorer
    import random
    import subprocess

    lmplz = DEFAULT_KENLM_BIN / "lmplz"
    if not lmplz.is_file():
        pytest.skip("KenLM binaries not built")
    rng = random.Random(0)
    corpus = tmp_path / "train.txt"
    corpus.write_text("".join("7 " + " ".join(str(rng.randint(8, 40)) for _ in range(rng.randint(3, 20))) + "\n"
                              for _ in range(2000)))
    arpa = tmp_path / "model.arpa"
    subprocess.run([str(lmplz), "-o", "5", "-S", "100M", "-T", str(tmp_path), "--discount_fallback",
                    "--text", str(corpus), "--arpa", str(arpa)], check=True, capture_output=True)
    scorer = KenLMScorer(arpa)
    targets = [7, 12, 30, 2]
    mine, oov = scorer.target_log_probs(targets)

    null, sentence, out = kenlm.State(), kenlm.State(), kenlm.State()
    scorer.model.NullContextWrite(null)
    scorer.model.BeginSentenceWrite(sentence)
    assert oov == 0
    assert mine[0] == pytest.approx(scorer.model.BaseScore(null, "7", out) * LN10)
    assert mine[0] < scorer.model.BaseScore(sentence, "7", out) * LN10  # <s> would make 7 near-certain.
    assert np.isfinite(mine).all() and mine[-1] < 0  # Final EOS scored as </s>.
