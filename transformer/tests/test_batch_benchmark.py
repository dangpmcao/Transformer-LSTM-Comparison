"""Invented timing fixtures exercise selection; they are not GPU measurements."""

from copy import deepcopy

import torch

from transformer.batch_benchmark import recommend


def timing_rows():
    return [{"size": size, "microbatch_windows": batch, "status": "complete",
             "memory_headroom_ok": True, "wall_seconds_per_million_tokens": seconds}
            for batch, seconds in ((16, 20), (32, 10), (64, 15)) for size in ("s", "m", "l")]


def test_choose_measured_speed_instead_of_largest_batch():
    assert recommend(timing_rows())["microbatch_windows"] == 32


def test_memory_failure_or_missing_size_disqualifies_common_batch():
    rows = timing_rows()
    rows[5]["memory_headroom_ok"] = False
    assert recommend(rows)["microbatch_windows"] == 64
    rows[8]["status"] = "FAILED"
    assert recommend(rows)["microbatch_windows"] == 16
    assert recommend(rows[1:]) == {}


def test_validation_losses_cannot_affect_batch_selection():
    rows = timing_rows()
    modified = deepcopy(rows)
    for row in modified:
        row["validation_nll"] = 1e10 if row["microbatch_windows"] == 32 else 0
    assert recommend(rows) == recommend(modified)


def test_microbatches_preserve_normalized_update_with_padding_and_partial_batches():
    """The same effective batch gives the same float32 update up to rounding.

    Include an epoch partial batch, right padding, and a truncated final token
    budget. Dropout is zero, as selected by the actual S tuning study.
    """
    from transformer import train
    from transformer.model import TransformerConfig, TransformerLM
    from transformer.tests.test_training import SmallWindows
    from dataset.model_inputs import create_batch_loader, compute_masked_cross_entropy, limit_loss_mask_to_budget

    class Windows(SmallWindows):
        def __len__(self):
            return 70

        def __getitem__(self, index):
            return super().__getitem__(index % 2)

    states = []
    config = {"training": {"optimizer": {"weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8}}}
    for microbatch in (16, 32, 64):
        train.set_seed(13)
        model = TransformerLM(TransformerConfig(vocab_size=16, max_positions=8,
                                                layers=1, width=16, heads=4, dropout=0))
        optimizer = train.make_optimizer(model, config)
        for group in optimizer.param_groups:
            group["lr"] = 0.001
        consumed = pending_tokens = pending_windows = cursor = 0
        for batch in create_batch_loader(Windows(None, "train"), batch_size=microbatch,
                                         training=True, seed=13, epoch=0):
            mask = limit_loss_mask_to_budget(batch["loss_mask"], 275 - consumed - pending_tokens)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"])
            loss, count = compute_masked_cross_entropy(logits, batch["target_ids"], mask)
            loss.backward()
            pending_tokens += int(count)
            pending_windows += len(batch["input_ids"])
            cursor += len(batch["input_ids"])
            if pending_windows < 64 and cursor != 70 and consumed + pending_tokens != 275:
                continue
            for parameter in model.parameters():
                parameter.grad.div_(pending_tokens)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            consumed += pending_tokens
            pending_tokens = pending_windows = 0
        assert consumed == 275
        states.append(model.state_dict())
    for state in states[1:]:
        for key in state:
            torch.testing.assert_close(state[key], states[0][key], atol=5e-7, rtol=2e-5)
