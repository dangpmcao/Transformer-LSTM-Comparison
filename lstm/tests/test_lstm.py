"""LSTM: executable parameter matching, tying, fresh state per window, init, and shared training."""

from argparse import Namespace

import pytest
import torch

from comparison.protocol import load_protocol
from lstm.model import MODEL_SIZES, PROTOCOL_SIZE, LSTMConfig, LSTMLM
from transformer import train
from transformer.model import TransformerConfig, TransformerLM
from transformer.tests.test_training import SmallWindows, workspace_tmp  # noqa: F401  (shared fixtures)

PROTOCOL = load_protocol()


@pytest.mark.parametrize("size", sorted(MODEL_SIZES))
def test_non_embedding_parameters_match_transformer_within_protocol_tolerance(size):
    """TA #2: validate the parameter table in executable definitions, adapter included."""
    lstm = LSTMLM(LSTMConfig.for_size(size)).parameter_counts()
    reference = TransformerLM(TransformerConfig.for_size(PROTOCOL_SIZE[size])).parameter_counts()
    relative = (lstm["non_embedding"] - reference["non_embedding"]) / reference["non_embedding"]
    assert abs(relative) <= PROTOCOL["matching"]["non_embedding_relative_tolerance"]
    assert abs(relative) <= 0.002  # The chosen widths are much tighter than the 5% tolerance.
    assert lstm["token_embedding"] == reference["token_embedding"]  # Same d, same vocabulary.


def test_parameter_count_formula_includes_adapter_and_counts_tied_head_once():
    config = LSTMConfig.for_size("s")
    d, h, layers, vocab = config.width, config.hidden, config.layers, config.vocab_size
    recurrent = sum(4 * h * ((d if layer == 0 else h) + h) + 8 * h for layer in range(layers))
    counts = LSTMLM(config).parameter_counts()
    assert counts["non_embedding"] == recurrent + h * d + d
    assert counts["total"] == counts["non_embedding"] + vocab * d


def test_output_head_is_tied_and_adapter_maps_hidden_to_embedding_width():
    model = LSTMLM(LSTMConfig.for_size("s"))
    assert model.output_head.weight is model.token_embedding.weight
    assert model.output_adapter.in_features == 472 and model.output_adapter.out_features == 256
    assert model.lstm.proj_size == 0  # Classic LSTM: no projection inside the recurrence.


def test_initialization_follows_protocol():
    model = LSTMLM(LSTMConfig.for_size("m_deep"))
    h = model.config.hidden
    for layer in range(model.config.layers):
        bias_ih, bias_hh = getattr(model.lstm, f"bias_ih_l{layer}"), getattr(model.lstm, f"bias_hh_l{layer}")
        assert torch.all(bias_ih[h:2 * h] == 1) and torch.all(bias_ih[:h] == 0) and torch.all(bias_hh == 0)
    assert torch.all(model.token_embedding.weight[0] == 0)
    assert abs(model.lstm.weight_hh_l0.std().item() - 0.02) < 0.002


def tiny():
    torch.manual_seed(0)
    return LSTMLM(LSTMConfig(vocab_size=32, max_positions=16, layers=2, width=8, hidden=12, dropout=0.0)).eval()


def test_state_is_fresh_every_call_and_rows_are_independent():
    model, ids = tiny(), torch.randint(4, 32, (2, 10))
    first = model(ids)
    model(torch.randint(4, 32, (2, 10)))  # A different call in between must not leak state.
    torch.testing.assert_close(model(ids), first, rtol=0, atol=0)
    torch.testing.assert_close(model(ids[1:]), first[1:], rtol=1e-6, atol=1e-6)


def test_right_padding_cannot_change_valid_predictions():
    model, ids = tiny(), torch.randint(4, 32, (1, 10))
    padded = ids.clone()
    padded[0, 6:] = 0
    torch.testing.assert_close(model(padded)[:, :6], model(ids)[:, :6], rtol=0, atol=0)


def test_shared_trainer_runs_lstm_and_records_architecture(workspace_tmp, monkeypatch):  # noqa: F811
    protocol = {
        "inputs": {"vocab_size": 16, "tokenizer_sha256": "fixture"}, "windows": {"targets": 8},
        "matching": {"embedding_widths": {"s": 8, "m": 8}},
        "training": {"effective_batch_windows": 64,
                     "optimizer": {"weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
                     "scheduler": {"warmup_fraction": 0.01, "minimum_lr_ratio": 0.1},
                     "gradient_clip_global_norm": 1.0},
        "selection": {"validation_every_prediction_tokens": 5},
    }
    monkeypatch.setattr(train, "load_protocol", lambda: protocol)
    monkeypatch.setattr(train, "load_tokenizer", lambda path: None)
    monkeypatch.setattr(train, "NextTokenWindowDataset", SmallWindows)
    monkeypatch.setattr(train, "sha256_file", lambda path: "fixture")
    monkeypatch.setattr(LSTMConfig, "for_size", staticmethod(
        lambda size, dropout: LSTMConfig(vocab_size=16, max_positions=8, layers=2, width=8, hidden=12,
                                         dropout=dropout)))
    args = Namespace(mode="exploratory", architecture="lstm", size="m_deep", token_dir=workspace_tmp / "tokens",
                     tokenizer_dir=workspace_tmp / "tokenizer", snapshot_file=workspace_tmp / "lock",
                     run_dir=workspace_tmp / "run", budget_tokens=10, microbatch_windows=2,
                     learning_rate=0.001, dropout=0.1, seed=101, save_every_updates=1, resume=False, device="cpu")
    summary = train.run(args)
    assert summary["architecture"] == "lstm" and summary["consumed_tokens"] == 10
    saved = torch.load(workspace_tmp / "run/final.pt", weights_only=False)
    assert saved["identity"]["architecture"] == "lstm" and saved["identity"]["model"]["hidden"] == 12
    with pytest.raises(ValueError):
        train.build_model_config("transformer", "m_deep", 0.1)
