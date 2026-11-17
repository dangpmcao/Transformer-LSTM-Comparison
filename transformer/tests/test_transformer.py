import math

import pytest
import torch

from dataset.model_inputs import compute_masked_cross_entropy
from transformer.model import TransformerConfig, TransformerLM
from transformer.train import learning_rate


def tiny_model(dropout=0.0):
    return TransformerLM(TransformerConfig(
        vocab_size=32, max_positions=8, layers=2, width=16, heads=4,
        dropout=dropout,
    ))


@pytest.mark.parametrize("batch_size", [1, 3])
def test_shape_tying_and_finite_gradients(batch_size):
    model = tiny_model()
    inputs = torch.randint(4, 32, (batch_size, 8))
    inputs[:, 0] = 1
    targets = torch.randint(4, 32, (batch_size, 8))
    mask = torch.ones_like(inputs, dtype=torch.bool)
    logits = model(inputs, mask)
    loss, count = compute_masked_cross_entropy(logits, targets, mask)
    loss.div(count).backward()
    assert logits.shape == (batch_size, 8, 32)
    assert model.output_head.weight is model.token_embedding.weight
    assert model.parameter_counts()["total"] == sum(p.numel() for p in model.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_future_and_padding_cannot_change_valid_logits():
    model = tiny_model().eval()
    original = torch.tensor([[1, 7, 8, 9, 0, 0, 0, 0]])
    changed_future = torch.tensor([[1, 7, 8, 20, 21, 22, 23, 24]])
    changed_padding = torch.tensor([[1, 7, 8, 9, 21, 22, 23, 24]])
    valid = torch.tensor([[True, True, True, True, False, False, False, False]])
    with torch.no_grad():
        first = model(original, valid)
        future = model(changed_future, valid)
        padding = model(changed_padding, valid)
        single = model(original[:, :4], valid[:, :4])
    torch.testing.assert_close(first[:, :3], future[:, :3])
    torch.testing.assert_close(first[:, :4], padding[:, :4])
    torch.testing.assert_close(first[:, :4], single)
    targets = torch.tensor([[7, 8, 9, 2, 0, 0, 0, 0]])
    loss_a, _ = compute_masked_cross_entropy(first, targets, valid)
    loss_b, _ = compute_masked_cross_entropy(padding, targets, valid)
    torch.testing.assert_close(loss_a, loss_b)


def test_right_padding_is_required():
    model = tiny_model()
    with pytest.raises(ValueError, match="right-padded"):
        model(torch.tensor([[1, 5, 0, 6]]), torch.tensor([[True, True, False, True]]))


def test_protocol_sizes_and_schedule():
    expected = {"s": (4, 256, 4), "m": (6, 384, 6), "l": (8, 640, 10)}
    expected_counts = {
        "s": (4_194_304, 3_290_624, 7_484_928),
        "m": (6_291_456, 10_844_160, 17_135_616),
        "l": (10_485_760, 39_717_120, 50_202_880),
    }
    for size, shape in expected.items():
        config = TransformerConfig.for_size(size)
        assert (config.layers, config.width, config.heads) == shape
        counts = TransformerLM(config).parameter_counts()
        assert tuple(counts.values()) == expected_counts[size]
    peak = 0.001
    assert learning_rate(10, 1000, peak, 0.01, 0.1) == pytest.approx(peak)
    assert learning_rate(1000, 1000, peak, 0.01, 0.1) == pytest.approx(peak * 0.1)
    assert math.isfinite(learning_rate(500, 1000, peak, 0.01, 0.1))


def test_positions_restart_for_each_window():
    model = tiny_model().eval()
    inputs = torch.tensor([[1, 5, 6], [1, 7, 8]])
    positions = torch.arange(3).expand(2, -1)
    with torch.no_grad():
        torch.testing.assert_close(model(inputs), model(inputs, position_ids=positions))
    with pytest.raises(ValueError, match="restart at zero"):
        model(inputs, position_ids=positions + 1)
