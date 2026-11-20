"""LSTM final evaluation: fresh state per input, padding invariance, same-target shortcut."""

from pathlib import Path

import numpy as np
import pytest
import torch

from comparison.same_target import FULL, load_rules
from lstm.evaluate import run_label, score_same_target, sequences_nll, token_nll
from lstm.model import LSTMConfig, LSTMLM

EOS = 2


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    lm = LSTMLM(LSTMConfig(vocab_size=64, layers=2, width=8, hidden=12, dropout=0.1))
    return lm.eval()


def reference_nll(model, targets):
    """Direct computation: one sequence, [BOS] + targets[:-1] -> targets."""
    inputs = torch.tensor([[1] + targets[:-1]])
    with torch.no_grad():
        log_probs = torch.log_softmax(model(inputs).double(), -1)[0]
    return -log_probs[torch.arange(len(targets)), torch.tensor(targets)].numpy()


def test_sequences_are_scored_independently_of_batch_and_padding(model):
    rng = np.random.default_rng(0)
    sequences = [rng.integers(4, 64, length).tolist() for length in (3, 17, 9, 30, 1)]
    batched = sequences_nll(model, sequences, "cpu", batch_size=4)
    for sequence, values in zip(sequences, batched, strict=True):
        assert values.shape == (len(sequence),)
        assert np.allclose(values, reference_nll(model, sequence), atol=1e-5)


def test_token_nll_marks_padding_as_nan(model):
    from comparison.blimp_scorer import make_window_batch

    nll = token_nll(model, make_window_batch([[5, 6, 7], [8]]))
    assert np.isfinite(nll[0]).all() and np.isfinite(nll[1, 0]) and np.isnan(nll[1, 1:]).all()


def test_eval_mode_is_deterministic(model):
    sequence = [5, 9, 11, 4, 60]
    first, second = (sequences_nll(model, [sequence], "cpu", 1)[0] for _ in range(2))
    assert np.array_equal(first, second)


def test_same_target_full_shortcut_matches_explicit_prefixes(model):
    rules = load_rules()
    rng = np.random.default_rng(1)
    tokens = np.concatenate([rng.integers(4, 64, 512), [EOS]]).astype("<u2")
    offsets = np.array([0, 513], dtype="<u8")
    positions = (385, 431, 512)
    targets = np.array([[0, 0, p, tokens[p - 1]] for p in positions], dtype="<i8")
    nll, check = score_same_target(model, targets, tokens, offsets, rules, device="cpu", batch_size=2)
    assert check["status"] == "PASS" and check["rows"] == 3
    for row, position in enumerate(positions):
        window = tokens[:512].astype(int).tolist()
        assert nll[FULL][row] == pytest.approx(reference_nll(model, window[:position])[-1], abs=1e-5)
        k4 = window[position - 5:position]  # 4 history tokens + target
        assert nll["k4"][row] == pytest.approx(reference_nll(model, k4)[-1], abs=1e-5)
    assert set(nll) == set(rules.conditions)


def test_run_label_names_campaign_size_and_seed():
    assert run_label(Path("lstm/runs/production/m_deep/seed13")) == "lstm_production_m_deep_seed13"
    assert run_label(Path("lstm/runs/sensitivity_mdeep_lr0.01/m_deep/seed37")) == \
        "lstm_sensitivity_mdeep_lr0.01_m_deep_seed37"
