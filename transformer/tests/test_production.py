"""Synthetic campaign tests, including real CPU training and interrupted resume."""

from argparse import Namespace
from copy import deepcopy
import fcntl
from pathlib import Path
import subprocess

import pytest
import torch

from comparison.protocol import freeze, check_lock
from comparison.tests.test_protocol import production_shape_fixture
from dataset.common import read_json, sha256_file, write_json
from transformer import production, train
from transformer.model import TransformerConfig
from transformer.tests.test_training import SmallWindows
from transformer.tests.test_tuning import study, fake_complete


@pytest.fixture
def campaign(study, monkeypatch):
    source, settings = study
    lock = read_json(source / "protocol.lock.json")
    lock["configuration"]["inputs"]["tokenizer_sha256"] = "fixture"
    write_json(source / "protocol.lock.json", lock)
    settings["snapshot_sha256"] = sha256_file(source / "protocol.lock.json")
    write_json(source / "study.json", settings)
    for trial in settings["trials"]:
        fake_complete(source, settings, trial, nll=3.0 if trial["index"] == 4 else 4.0)
    ready = deepcopy(lock)
    ready["production_ready"] = True
    ready["configuration"]["budget"].update(
        exact_prediction_tokens=10, tuning_prediction_tokens=10,
        microbatch_windows=2, gradient_accumulation_steps=32)
    snapshot = source / "ready.json"
    write_json(snapshot, ready)
    monkeypatch.setattr(production, "check_lock", lambda *args, **kwargs: ready)
    root = source / "production"
    production.prepare(root, tuning_root=source, snapshot=snapshot)
    return root, source, snapshot, ready


def test_nine_distinct_runs_and_selected_settings_are_frozen(campaign):
    root, source, snapshot, ready = campaign
    settings = production.verify(root)
    assert {(row["size"], row["seed"]) for row in settings["runs"]} == {
        (size, seed) for size in ("s", "m", "l") for seed in (13, 37, 73)}
    assert [row["size"] for row in settings["runs"]][:3] == ["l"] * 3
    assert settings["learning_rate"] == 0.001 and settings["dropout"] == 0
    assert production.prepare(root, tuning_root=source, snapshot=snapshot) == settings
    assert production.collect(root)["completed_runs"] == 0
    command = production.trainer_command(root, settings, settings["runs"][0], "cpu")
    assert command[command.index("--mode") + 1] == "production"
    assert "--budget-tokens" not in command and "--microbatch-windows" not in command
    ready["configuration"]["budget"]["tuning_prediction_tokens"] = 20
    with pytest.raises(ValueError, match="tuning budget differs"):
        production.prepare(root, tuning_root=source, snapshot=snapshot)


def test_production_cannot_bypass_real_prepared_lock(production_shape_fixture, tmp_path, monkeypatch):
    artifacts, tokens, protocol = production_shape_fixture
    snapshot = tmp_path / "prepared.json"
    freeze(tokens, artifacts=artifacts, protocol=protocol, output=snapshot)
    monkeypatch.setattr(production, "selected_tuning", lambda root: ({}, {
        "token_dir": str(tokens), "tokenizer_dir": str(artifacts)}))
    monkeypatch.setattr(production, "check_lock", lambda *args, **kwargs:
                        check_lock(*args, **kwargs, protocol=protocol))
    with pytest.raises(ValueError, match="Production is blocked"):
        production.prepare(tmp_path / "production", tuning_root=tmp_path, snapshot=snapshot)
    assert not (tmp_path / "production").exists()


def test_changed_manifest_and_second_writer_are_rejected(campaign):
    root, _, _, _ = campaign
    with (root / "run_00.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already has a writer"):
            production.run_task(root, 0, device="cpu")
    settings = read_json(root / "campaign.json")
    settings["runs"][0]["seed"] = 101
    write_json(root / "campaign.json", settings)
    with pytest.raises(ValueError, match="settings differ"):
        production.verify(root)


@pytest.mark.parametrize("standalone", [False, True])
def test_small_production_train_resume_exact_budget_and_skip_completed(campaign, monkeypatch, standalone):
    root, source, _, ready = campaign
    if standalone:
        root = source / "standalone"
        production.prepare_standalone(root, tuning_root=source, budget_tokens=10, microbatch_windows=2)
    config = deepcopy(ready["configuration"])
    config["inputs"]["vocab_size"] = 16
    config["windows"]["targets"] = 8
    config["matching"]["embedding_widths"]["l"] = 16
    config["selection"]["validation_every_prediction_tokens"] = 5
    monkeypatch.setattr(train, "check_lock", lambda *args, **kwargs: {**ready, "configuration": config})
    monkeypatch.setattr(train, "load_protocol", lambda: config)
    monkeypatch.setattr(train, "load_tokenizer", lambda path: None)
    monkeypatch.setattr(train, "NextTokenWindowDataset", SmallWindows)
    monkeypatch.setattr(TransformerConfig, "for_size", staticmethod(
        lambda size, dropout: TransformerConfig(vocab_size=16, max_positions=8,
                                                layers=1, width=16, heads=4, dropout=dropout)))
    settings = production.verify(root)
    task = settings["runs"][0]
    expected = production.expected_identity(root, settings, task)
    hashes = {"protocol.yaml": expected["protocol_sha256"],
              "protocol.lock.json": expected["snapshot_sha256"],
              "tokenizer.json": expected["tokenizer_sha256"],
              "model.py": expected["model_code_sha256"], "train.py": expected["train_code_sha256"]}
    monkeypatch.setattr(train, "sha256_file", lambda path:
                        expected["window_manifest_sha256"] if str(path).endswith("windows/manifest.json")
                        else expected["token_manifest_sha256"] if Path(path).name == "manifest.json"
                        else hashes[Path(path).name])
    original_save = train.save_torch
    calls = []

    def launch(command, **kwargs):
        calls.append(command)
        def value(flag):
            return command[command.index(flag) + 1]
        train.run(Namespace(
            mode=value("--mode"), size=value("--size"), seed=int(value("--seed")),
            snapshot_file=Path(value("--snapshot-file")), token_dir=Path(value("--token-dir")),
            tokenizer_dir=Path(value("--tokenizer-dir")), run_dir=Path(value("--run-dir")),
            budget_tokens=int(value("--budget-tokens")) if standalone else None,
            microbatch_windows=int(value("--microbatch-windows")) if standalone else None,
            learning_rate=float(value("--learning-rate")),
            dropout=float(value("--dropout")), save_every_updates=1,
            resume="--resume" in command, device="cpu"))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(production.subprocess, "run", launch)
    def interrupted_save(path, data):
        original_save(path, data)
        if path.name == "latest.pt" and data.get("updates") == 1:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(train, "save_torch", interrupted_save)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        production.run_task(root, 0, device="cpu")
    assert production.collect(root)["runs"][0]["status"] == "failed"
    monkeypatch.setattr(train, "save_torch", original_save)
    assert production.run_task(root, 0, device="cpu") == 0
    assert "--resume" in calls[-1]
    report = production.collect(root)
    assert report["completed_runs"] == 1 and report["status"] == "incomplete"
    assert report["runs"][0]["result"]["consumed_tokens"] == 10
    saved = torch.load(root / task["directory"] / "latest.pt", map_location="cpu", weights_only=False)
    assert saved["identity"]["seed"] == 13
    assert all(tensor.device.type == "cpu" and tensor.dtype == torch.float32
               and torch.isfinite(tensor).all() for tensor in saved["model"].values())
    assert production.run_task(root, 0, device="cpu") == 0 and len(calls) == 2
    saved_identity = read_json(root / task["directory"] / "run.json")
    saved_identity["identity"]["seed"] = 37
    write_json(root / task["directory"] / "run.json", saved_identity)
    with pytest.raises(ValueError, match="provenance"):
        production.collect(root)


def test_standalone_settings_are_explicit_and_cannot_be_changed(campaign):
    _, source, _, _ = campaign
    root = source / "standalone"
    kwargs = dict(tuning_root=source, budget_tokens=250_000_000, microbatch_windows=64)
    settings = production.prepare_standalone(root, **kwargs)
    assert settings["joint_production_ready"] is False
    assert settings["trainer_mode"] == "exploratory"
    assert settings["gradient_accumulation_steps"] == 1
    assert production.prepare_standalone(root, **kwargs) == settings
    assert production.verify(root) == settings
    assert len(settings["runs"]) == 9
    with pytest.raises(ValueError, match="settings differ"):
        production.prepare_standalone(root, **{**kwargs, "budget_tokens": 100})
    with pytest.raises(ValueError, match="Microbatch must divide"):
        production.prepare_standalone(source / "bad", **{**kwargs, "microbatch_windows": 3})
    execution = read_json(root / "execution.json")
    execution["budget_tokens"] = 100
    write_json(root / "execution.json", execution)
    with pytest.raises(ValueError, match="execution settings changed"):
        production.verify(root)
