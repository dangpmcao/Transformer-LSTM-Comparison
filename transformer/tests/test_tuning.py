"""Tuning orchestration tests; synthetic scores are never research results."""

from argparse import Namespace
import fcntl
import json
from pathlib import Path
import subprocess

import pytest
import torch

from comparison.protocol import load_protocol
from comparison.tests.test_protocol import production_shape_fixture
from dataset.common import read_json, sha256_file, write_json
from transformer import train, tuning
from transformer.model import TransformerConfig
from transformer.tests.test_training import SmallWindows


@pytest.fixture
def study(tmp_path, monkeypatch):
    config = load_protocol()
    lock = {"configuration": config, "protocol_sha256": "fixture",
            "token_manifest_sha256": "fixture", "window_manifest_sha256": "fixture"}
    write_json(tmp_path / "protocol.lock.json", lock)
    settings = {
        "size": "s", "budget_tokens": 10, "microbatch_windows": 2,
        "effective_batch_windows": 64, "seed": 101,
        "token_dir": str(tmp_path / "tokens"), "tokenizer_dir": str(tmp_path / "tokenizer"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "code_sha256": tuning.code_hashes(), "trials": tuning.trial_grid(config),
        "snapshot_sha256": sha256_file(tmp_path / "protocol.lock.json"),
    }
    write_json(tmp_path / "study.json", settings)
    monkeypatch.setattr(tuning, "check_lock", lambda *args, **kwargs: lock)
    return tmp_path, settings


def fake_complete(root, study, trial, nll=4.0):
    directory = root / trial["directory"]
    directory.mkdir(exist_ok=True)
    lock = read_json(root / "protocol.lock.json")
    identity = {key: study[key] for key in ("size", "seed", "budget_tokens", "microbatch_windows",
                                           "effective_batch_windows", "torch_version", "cuda_version")}
    identity.update({key: lock[key] for key in ("protocol_sha256", "token_manifest_sha256", "window_manifest_sha256")})
    identity.update({"mode": "exploratory", "learning_rate": trial["learning_rate"],
                     "model": {"dropout": trial["dropout"]},
                     "tokenizer_sha256": lock["configuration"]["inputs"]["tokenizer_sha256"],
                     "model_code_sha256": study["code_sha256"]["transformer/model.py"],
                     "train_code_sha256": study["code_sha256"]["transformer/train.py"]})
    write_json(directory / "run.json", {"identity": identity})
    write_json(directory / "summary.json", {"status": "complete", "consumed_tokens": study["budget_tokens"],
                                            "best_validation_nll": nll, "best_checkpoint_tokens": 8})
    for name in ("best.pt", "final.pt", "latest.pt"):
        (directory / name).write_bytes(b"synthetic fixture")


def test_preparation_freezes_real_pipeline_and_refuses_changed_budget(production_shape_fixture, tmp_path, monkeypatch):
    artifacts, tokens, protocol = production_shape_fixture
    monkeypatch.setattr(tuning, "DEFAULT_PROTOCOL", protocol)
    monkeypatch.setattr(tuning, "load_protocol", lambda: load_protocol(protocol))
    root = tmp_path / "study"
    settings = dict(budget_tokens=1000, microbatch_windows=16, token_dir=tokens, tokenizer_dir=artifacts)
    result = tuning.prepare(root, **settings)
    assert len(result["trials"]) == 12
    assert tuning.prepare(root, **settings) == result
    assert read_json(root / "protocol.lock.json")["production_ready"] is False
    with pytest.raises(ValueError, match="different settings"):
        tuning.prepare(root, **{**settings, "budget_tokens": 2000})


def test_incomplete_grid_never_selects_winner(study):
    root, settings = study
    fake_complete(root, settings, settings["trials"][0])
    result = tuning.collect(root)
    assert result["status"] == "incomplete"
    assert result["winner"] is None


def test_selection_uses_validation_then_grid_order_and_keeps_failures(study):
    root, settings = study
    for trial in settings["trials"][:-1]:
        fake_complete(root, settings, trial, nll=3.0 if trial["index"] in (2, 4) else 4.0)
    write_json(root / "attempts/trial_11_100.json", {"status": "failed", "returncode": 1})
    result = tuning.collect(root)
    assert result["status"] == "finished"
    assert result["winner"]["index"] == 2
    assert result["failed_trials"] == 1
    bad = root / "trial_02/run.json"
    changed = read_json(bad)
    changed["identity"]["seed"] = 13
    write_json(bad, changed)
    with pytest.raises(ValueError, match="provenance"):
        tuning.collect(root)


def test_command_resume_and_writer_lock(study):
    root, settings = study
    trial = settings["trials"][0]
    command = tuning.trainer_command(root, settings, trial, device="cpu")
    assert "--resume" not in command
    assert command[command.index("--learning-rate") + 1] == "0.0001"
    (root / trial["directory"]).mkdir()
    (root / trial["directory"] / "latest.pt").touch()
    assert tuning.trainer_command(root, settings, trial, device="cpu")[-1] == "--resume"
    with (root / "trial_00.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer"):
            tuning.run_trial(root, 0, device="cpu")


def test_subprocess_failures_retained_and_can_be_retried(study, monkeypatch):
    root, settings = study
    monkeypatch.setattr(tuning.subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, 1))
    assert tuning.run_trial(root, 0, device="cpu") == 1
    assert tuning.collect(root)["trials"][0]["status"] == "failed"
    def succeed(*args, **kwargs):
        fake_complete(root, settings, settings["trials"][0])
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(tuning.subprocess, "run", succeed)
    assert tuning.run_trial(root, 0, device="cpu") == 0
    assert len(list((root / "attempts").glob("*.json"))) == 2
    assert tuning.run_trial(root, 0, device="cpu") == 0  # Completed trial is not rerun.


def test_small_real_trainer_through_tuning_runner(study, monkeypatch):
    root, settings = study
    config = {
        "inputs": {"vocab_size": 16, "tokenizer_sha256": "fixture"}, "windows": {"targets": 8},
        "matching": {"embedding_widths": {"s": 16}},
        "training": {"effective_batch_windows": 64,
                     "optimizer": {"weight_decay": 0.1, "betas": [0.9, 0.95], "eps": 1e-8},
                     "scheduler": {"warmup_fraction": 0.01, "minimum_lr_ratio": 0.1},
                     "gradient_clip_global_norm": 1.0},
        "selection": {"validation_every_prediction_tokens": 5},
    }
    monkeypatch.setattr(train, "load_protocol", lambda: config)
    monkeypatch.setattr(train, "load_tokenizer", lambda path: None)
    monkeypatch.setattr(train, "NextTokenWindowDataset", SmallWindows)
    lock = read_json(root / "protocol.lock.json")
    lock["configuration"]["inputs"]["tokenizer_sha256"] = "fixture"
    write_json(root / "protocol.lock.json", lock)
    settings["snapshot_sha256"] = sha256_file(root / "protocol.lock.json")
    write_json(root / "study.json", settings)
    hashes = settings["code_sha256"]
    monkeypatch.setattr(train, "sha256_file", lambda path: hashes.get(f"transformer/{Path(path).name}", "fixture"))
    monkeypatch.setattr(TransformerConfig, "for_size", staticmethod(
        lambda size, dropout: TransformerConfig(vocab_size=16, max_positions=8,
                                                layers=1, width=16, heads=4, dropout=dropout)))
    def launch(command, **kwargs):
        def value(flag):
            return command[command.index(flag) + 1]
        train.run(Namespace(
            mode="exploratory", size="s", seed=int(value("--seed")),
            budget_tokens=int(value("--budget-tokens")), microbatch_windows=int(value("--microbatch-windows")),
            learning_rate=float(value("--learning-rate")), dropout=float(value("--dropout")),
            token_dir=Path(value("--token-dir")), tokenizer_dir=Path(value("--tokenizer-dir")),
            run_dir=Path(value("--run-dir")), snapshot_file=root / "protocol.lock.json",
            save_every_updates=25, resume="--resume" in command, device="cpu"))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(tuning.subprocess, "run", launch)
    assert tuning.run_trial(root, 0, device="cpu") == 0
    saved = torch.load(root / "trial_00/latest.pt", weights_only=False)
    assert saved["consumed"] == 10
    assert all(value.device.type == "cpu" and value.dtype == torch.float32 and torch.isfinite(value).all()
               for value in saved["model"].values())
    assert tuning.collect(root)["trials"][0]["status"] == "complete"
