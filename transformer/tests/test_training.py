from argparse import Namespace
from pathlib import Path
import shutil
import uuid

import pytest
import torch

from transformer import train
from transformer.model import TransformerConfig


class SmallWindows:
    def __init__(self, root, split):
        self.split = split
        self.manifest = {"tokenizer_sha256": "fixture"}

    def __len__(self):
        return 2

    def __getitem__(self, index):
        targets = torch.tensor([4 + index, 5 + index, 6 + index, 2, 0, 0, 0, 0])
        inputs = torch.tensor([1, 4 + index, 5 + index, 6 + index, 0, 0, 0, 0])
        valid = torch.tensor([True] * 4 + [False] * 4)
        return {
            "input_ids": inputs, "target_ids": targets,
            "loss_mask": valid, "attention_mask": valid.clone(),
            "position_ids": torch.arange(8), "window_id": f"{self.split}:{index}",
        }


@pytest.fixture
def workspace_tmp():
    parent = Path(__file__).resolve().parents[1] / ".test_tmp"
    parent.mkdir(exist_ok=True)
    path = parent / uuid.uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        if not path.resolve().is_relative_to(parent.resolve()):
            raise RuntimeError("Unsafe test cleanup path")
        shutil.rmtree(path)


def test_exact_budget_and_interrupted_resume_match_uninterrupted(workspace_tmp, monkeypatch):
    tmp_path = workspace_tmp
    protocol = {
        "inputs": {"vocab_size": 16, "tokenizer_sha256": "fixture"},
        "windows": {"targets": 8},
        "matching": {"embedding_widths": {"s": 16}},
        "training": {
            "effective_batch_windows": 64,
            "optimizer": {"weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
            "scheduler": {"warmup_fraction": 0.01, "minimum_lr_ratio": 0.1},
            "gradient_clip_global_norm": 1.0,
        },
        "selection": {"validation_every_prediction_tokens": 5},
    }
    monkeypatch.setattr(train, "load_protocol", lambda: protocol)
    monkeypatch.setattr(train, "load_tokenizer", lambda path: None)
    monkeypatch.setattr(train, "NextTokenWindowDataset", SmallWindows)
    monkeypatch.setattr(train, "sha256_file", lambda path: "fixture")
    monkeypatch.setattr(TransformerConfig, "for_size", staticmethod(
        lambda size, dropout: TransformerConfig(
            vocab_size=16, max_positions=8, layers=1, width=16, heads=4,
            dropout=dropout,
        ),
    ))

    def arguments(name, resume=False):
        return Namespace(
            mode="exploratory", size="s", token_dir=tmp_path / "tokens",
            tokenizer_dir=tmp_path / "tokenizer", snapshot_file=tmp_path / "lock",
            run_dir=tmp_path / name, budget_tokens=10, microbatch_windows=2,
            learning_rate=0.0003, dropout=0.1, seed=101,
            save_every_updates=1, resume=resume, device="cpu",
        )

    baseline = train.run(arguments("baseline"))
    original_save = train.save_torch

    def interrupt_after_saved_update(path, value):
        original_save(path, value)
        if path.name == "latest.pt" and value.get("updates") == 1:
            raise RuntimeError("simulated interruption after a full saved update")

    monkeypatch.setattr(train, "save_torch", interrupt_after_saved_update)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train.run(arguments("resumed"))
    monkeypatch.setattr(train, "save_torch", original_save)
    resumed = train.run(arguments("resumed", resume=True))
    assert baseline["consumed_tokens"] == resumed["consumed_tokens"] == 10
    assert baseline["updates"] == resumed["updates"] == 2
    first = torch.load(tmp_path / "baseline/final.pt", weights_only=False)
    second = torch.load(tmp_path / "resumed/final.pt", weights_only=False)
    for key in first["model"]:
        torch.testing.assert_close(first["model"][key], second["model"][key], rtol=0, atol=0)
