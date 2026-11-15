"""Pilot accounting tests use invented timings; the training smoke test is CPU-only."""

from argparse import Namespace
import json

import pytest
import torch

from dataset.common import write_json
from transformer import timing_pilot, train
from transformer.model import TransformerConfig
from transformer.tests.test_training import SmallWindows


def invented_events():
    events = []
    tokens, seconds = 0, 0.0
    # Warm-up; full; epoch-partial; full; budget-final.
    for index, (count, duration, windows) in enumerate(
            [(100, 1, 64), (200, 2, 64), (5, 1, 2), (300, 4, 64), (50, 1, 64)], start=1):
        tokens += count
        seconds += duration
        events.append({"event": "update", "updates": index, "consumed_tokens": tokens,
                       "prediction_tokens": count, "batch_windows": windows,
                       "train_seconds": seconds, "training_nll": 4.0,
                       "gradient_norm": 0.5, "learning_rate": 0.0003})
    return events


def summarize(events, minimum_updates=2):
    return timing_pilot.summarize_updates(events, warmup_updates=1,
                                         minimum_updates=minimum_updates,
                                         effective_batch=64, budget=655)


def test_throughput_uses_actual_tokens_and_excludes_warmup_partial_and_final():
    result = summarize(invented_events())
    assert result["measured_updates"] == 2
    assert result["measured_prediction_tokens"] == 500
    assert result["measured_training_seconds"] == 6
    assert result["prediction_tokens_per_second"] == pytest.approx(500 / 6)
    assert result["update_tokens_per_second_min"] == 75
    assert result["update_tokens_per_second_max"] == 100


@pytest.mark.parametrize("problem", ["too_short", "nonfinite", "counter_gap", "incomplete", "zero_time"])
def test_invalid_measurements_cannot_be_exported(problem):
    events = invented_events()
    if problem == "too_short":
        minimum = 3
    else:
        minimum = 2
    if problem == "nonfinite":
        events[1]["gradient_norm"] = float("nan")
    elif problem == "counter_gap":
        events[1]["prediction_tokens"] += 1
    elif problem == "incomplete":
        events.pop()
    elif problem == "zero_time":
        events[1]["train_seconds"] = events[0]["train_seconds"]
    with pytest.raises(ValueError):
        summarize(events, minimum)


def test_collect_measurement_preserves_pilot_field_names_and_checks_identity(tmp_path):
    args = Namespace(budget_tokens=655, microbatch_windows=16, seed=101,
                     learning_rate=0.0003, dropout=0.1, warmup_updates=1,
                     minimum_measured_updates=2)
    identity = {"size": "s", "mode": "exploratory", "budget_tokens": 655,
                "microbatch_windows": 16, "effective_batch_windows": 64,
                "seed": 101, "learning_rate": 0.0003, "model": {"dropout": 0.1},
                "protocol_sha256": "synthetic"}
    write_json(tmp_path / "run.json", {"identity": identity})
    (tmp_path / "latest.pt").write_bytes(b"synthetic checkpoint")
    summary = {"status": "complete", "mode": "exploratory", "consumed_tokens": 655,
               "latest_checkpoint_bytes": (tmp_path / "latest.pt").stat().st_size,
               "peak_cuda_bytes": 123456, "parameters": {"total": 123},
               "prediction_tokens_per_second": 655 / 9}
    write_json(tmp_path / "summary.json", summary)
    validations = [{"event": "validation", "consumed_tokens": tokens,
                    "seconds": 0.25, "tokens": 255, "nll": 4.0} for tokens in (0, 655)]
    events = [validations[0], *invented_events(), validations[1]]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in events))
    row = timing_pilot.collect_measurement(
        tmp_path, args=args, size="s", expected_identity={"protocol_sha256": "synthetic"},
        wall_seconds=12.0,
    )
    assert row["architecture"] == "transformer"
    assert row["peak_memory_bytes"] == 123456
    assert row["validation_seconds"] == 0.25
    assert row["checkpoint_bytes"] == len(b"synthetic checkpoint")
    assert row["prediction_tokens_per_second"] == pytest.approx(500 / 6)
    with pytest.raises(ValueError, match="identity"):
        timing_pilot.collect_measurement(tmp_path, args=args, size="m",
                                         expected_identity={}, wall_seconds=12.0)
    summary["peak_cuda_bytes"] = None
    write_json(tmp_path / "summary.json", summary)
    with pytest.raises(ValueError, match="peak CUDA"):
        timing_pilot.collect_measurement(tmp_path, args=args, size="s",
                                         expected_identity={}, wall_seconds=12.0)


def test_microbatch16_cpu_smoke_with_real_trainer_and_partial_epoch(tmp_path, monkeypatch):
    class PilotWindows(SmallWindows):
        def __len__(self):
            return 130

        def __getitem__(self, index):
            return super().__getitem__(index % 2)

    protocol = {
        "inputs": {"vocab_size": 16, "tokenizer_sha256": "fixture"},
        "windows": {"targets": 8}, "matching": {"embedding_widths": {"s": 16}},
        "training": {"effective_batch_windows": 64,
                     "optimizer": {"weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
                     "scheduler": {"warmup_fraction": 0.01, "minimum_lr_ratio": 0.1},
                     "gradient_clip_global_norm": 1.0},
        "selection": {"validation_every_prediction_tokens": 500},
    }
    monkeypatch.setattr(train, "load_protocol", lambda: protocol)
    monkeypatch.setattr(train, "load_tokenizer", lambda path: None)
    monkeypatch.setattr(train, "NextTokenWindowDataset", PilotWindows)
    monkeypatch.setattr(train, "sha256_file", lambda path: "fixture")
    monkeypatch.setattr(TransformerConfig, "for_size", staticmethod(
        lambda size, dropout: TransformerConfig(vocab_size=16, max_positions=8,
                                                layers=1, width=16, heads=4, dropout=dropout)))
    run_dir = tmp_path / "cpu_smoke"
    summary = train.run(Namespace(
        mode="exploratory", size="s", token_dir=tmp_path / "tokens",
        tokenizer_dir=tmp_path / "tokenizer", snapshot_file=tmp_path / "lock",
        run_dir=run_dir, budget_tokens=1000, microbatch_windows=16,
        learning_rate=0.0003, dropout=0.1, seed=101, save_every_updates=25,
        resume=False, device="cpu",
    ))
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    result = timing_pilot.summarize_updates(events, warmup_updates=1,
                                           minimum_updates=2, effective_batch=64, budget=1000)
    assert summary["consumed_tokens"] == 1000
    assert result["measured_updates"] == 2
    assert result["measured_prediction_tokens"] == 512
    checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
    assert checkpoint["consumed"] == 1000
    assert checkpoint["identity"]["microbatch_windows"] == 16
    assert checkpoint["identity"]["effective_batch_windows"] == 64
    assert all(value.dtype == torch.float32 and value.device.type == "cpu"
               and torch.isfinite(value).all() for value in checkpoint["model"].values())
