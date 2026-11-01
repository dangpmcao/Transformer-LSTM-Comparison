"""Protocol validation and production gates; all pilot records here are invented fixtures."""

import hashlib
import json
import random
import string

import pytest
import yaml

from comparison.protocol import apply_pilot, check_lock, freeze, load_protocol
from dataset.common import SOURCE_LOCK, file_record, read_json, write_json, sha256_file
from dataset.encode import encode
from dataset.windows import build_windows
from tokenizer.core import DEFAULT_CONFIG
from tokenizer.train import train


def pilot_fixture():
    return {
        "status": "PASS", "hardware": "SYNTHETIC TEST FIXTURE - NOT REAL TIMING",
        "settings": {"exact_prediction_tokens": 250000000, "tuning_prediction_tokens": 10000000,
                     "microbatch_windows": 8, "gradient_accumulation_steps": 8},
        "measurements": [{"architecture": a, "size": s, "prediction_tokens_per_second": 1000,
                          "peak_memory_bytes": 1000, "validation_seconds": 1, "checkpoint_bytes": 1000}
                         for a in ("transformer", "lstm") for s in ("s", "m", "l")],
    }


def test_protocol_and_deferred_pilot_settings():
    config = load_protocol()
    assert config["training"]["seeds"] == [13, 37, 73]
    assert config["budget"]["exact_prediction_tokens"] is None
    ready = apply_pilot(config, pilot_fixture())
    assert ready["budget"]["exact_prediction_tokens"] == 250000000
    assert config["budget"]["exact_prediction_tokens"] is None
    assert ready["stage"] == "production_ready_after_pilot"


@pytest.mark.parametrize("failure", ["missing_hardware", "missing_measurement", "bad_budget", "batch_mismatch", "nonfinite"])
def test_invalid_pilot_is_rejected(failure):
    pilot = pilot_fixture()
    if failure == "missing_hardware":
        pilot.pop("hardware")
    elif failure == "missing_measurement":
        pilot["measurements"].pop()
    elif failure == "bad_budget":
        pilot["settings"]["exact_prediction_tokens"] = True
    elif failure == "batch_mismatch":
        pilot["settings"]["microbatch_windows"] = 7
    else:
        pilot["measurements"][0]["prediction_tokens_per_second"] = float("nan")
    with pytest.raises(ValueError):
        apply_pilot(load_protocol(), pilot)


@pytest.mark.parametrize("section,key,value", [
    ("windows", "stride", 256), ("windows", "bos", "article_only"),
    ("training", "tuning_seed", 13), ("selection", "split", "test"),
    ("budget", "exact_prediction_tokens", 250000000),
])
def test_protocol_cannot_silently_break_shared_contract(tmp_path, section, key, value):
    config = load_protocol()
    config[section][key] = value
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_protocol(path)


@pytest.fixture(scope="module")
def production_shape_fixture(tmp_path_factory):
    """Synthetic 16,384-ID vocabulary; no production dataset or artifacts are read."""
    root = tmp_path_factory.mktemp("stage3-protocol")
    data = root / "data"
    articles = data / "processed/articles"
    articles.mkdir(parents=True)
    rng = random.Random(7)
    words = ["".join(rng.choices(string.ascii_letters, k=12)) for _ in range(9000)]
    train_text = (" ".join(words) + "\n") * 2
    splits = {}
    for split, text in (("train", train_text), ("validation", "Invented validation."), ("test", "Invented test.")):
        path = articles / f"{split}.jsonl"
        write_json_line = {"split": split, "article_id": f"{split}:000000", "text": text,
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        path.write_text(json.dumps(write_json_line) + "\n")
        splits[split] = {"articles": 1, "text_bytes": len(text.encode()), "file": file_record(path, data)}
    write_json(data / "manifest.json", {"sources": {"wikitext": read_json(SOURCE_LOCK)["wikitext"]}, "wikitext_splits": splits})
    write_json(data / "validation.json", {"status": "PASS", "manifest_sha256": sha256_file(data / "manifest.json")})
    artifacts = root / "artifacts"
    train(data, artifacts, DEFAULT_CONFIG)
    tokens = data / "processed/tokens"
    encode(data, artifacts, tokens)
    build_windows(tokens)
    protocol = load_protocol()
    protocol["inputs"]["tokenizer_sha256"] = read_json(artifacts / "manifest.json")["tokenizer_sha256"]
    path = root / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol))
    return artifacts, tokens, path


def test_prepared_lock_blocks_production_and_binds_inputs(production_shape_fixture, tmp_path):
    artifacts, tokens, protocol = production_shape_fixture
    output = tmp_path / "prepared.json"
    lock = freeze(tokens, artifacts=artifacts, protocol=protocol, output=output)
    assert lock["production_ready"] is False
    assert check_lock(output, tokens, protocol=protocol, artifacts=artifacts)["production_ready"] is False
    with pytest.raises(ValueError, match="Production is blocked"):
        check_lock(output, tokens, protocol=protocol, artifacts=artifacts, require_production=True)
    with pytest.raises(FileExistsError):
        freeze(tokens, artifacts=artifacts, protocol=protocol, output=output)
    changed = tmp_path / "changed.yaml"
    changed.write_text(protocol.read_text() + "\n# changed\n")
    with pytest.raises(ValueError, match="stale"):
        check_lock(output, tokens, protocol=changed, artifacts=artifacts)


def test_pilot_lock_can_pass_gate_with_recorded_evidence(production_shape_fixture, tmp_path):
    artifacts, tokens, protocol = production_shape_fixture
    pilot = tmp_path / "pilot.json"
    evidence = pilot_fixture()
    evidence.update({"protocol_sha256": sha256_file(protocol),
                     "token_manifest_sha256": sha256_file(tokens / "manifest.json"),
                     "window_manifest_sha256": sha256_file(tokens / "windows/manifest.json")})
    write_json(pilot, evidence)
    output = tmp_path / "ready.json"
    freeze(tokens, artifacts=artifacts, protocol=protocol, output=output, pilot_path=pilot)
    assert check_lock(output, tokens, protocol=protocol, artifacts=artifacts, require_production=True)["production_ready"] is True
