"""Synthetic tests only: no production tokenizer or dataset outputs are created."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from tokenizers import Tokenizer

from dataset.common import SOURCE_LOCK, file_record, read_json, sha256_file, write_json
from tokenizer.check import check
from tokenizer.core import (
    DEFAULT_CONFIG, PROBES, SPECIAL_IDS, article_batches, build_tokenizer,
    encode_text, load_tokenizer, validate_config, verify_structure,
)
from tokenizer.train import train


@pytest.fixture
def fixture_data(tmp_path):
    """Tiny local inputs with the Stage 1 manifest interface; never real WikiText."""
    root = tmp_path / "data"
    directory = root / "processed/articles"
    directory.mkdir(parents=True)
    summaries = {}
    train_text = (
        " = Synthetic training article = \n The quick brown fox jumps over the lazy dog. "
        "Research grade language models predict tokens from shared contexts. "
        "Café 日本語 😀 literal <bos> <pad> <eos> <unk>\n"
    )
    for split, texts in {
        "train": [train_text * 5, train_text * 4],
        "validation": ["supersecretvalidationsentinel " * 20],
        "test": ["supersecrettestsentinel " * 20],
    }.items():
        path = directory / f"{split}.jsonl"
        rows = [{"article_id": f"{split}:{i:06d}", "split": split, "text": text,
                 "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
                for i, text in enumerate(texts)]
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        summaries[split] = {"articles": len(rows), "text_bytes": sum(len(t.encode()) for t in texts),
                            "file": file_record(path, root)}
    # The locked source interface is reproduced for unit testing; synthetic content
    # and this deliberately tiny fixture are never published as research results.
    write_json(root / "manifest.json", {
        "test_fixture": True, "sources": {"wikitext": read_json(SOURCE_LOCK)["wikitext"]},
        "wikitext_splits": summaries,
    })
    write_json(root / "validation.json", {"status": "PASS", "manifest_sha256": sha256_file(root / "manifest.json")})
    config = read_json(DEFAULT_CONFIG)
    config.update(vocab_size=300, batch_size=2)
    config_path = tmp_path / "config.json"
    write_json(config_path, config)
    return root, config_path, tmp_path / "artifacts"


@pytest.fixture
def bundle(fixture_data):
    root, config, output = fixture_data
    train(root, output, config)
    return root, config, output


def test_complete_synthetic_training_and_statistics(bundle):
    root, _, output = bundle
    tokenizer = load_tokenizer(output)
    manifest = read_json(output / "manifest.json")
    checks = read_json(output / "checks.json")
    assert tokenizer.get_vocab_size() == 300
    assert manifest["training_articles"] == 2
    assert manifest["training_input"]["path"] == "processed/articles/train.jsonl"
    assert checks["status"] == "PASS" and checks["every_article_round_trip"]
    assert checks["blimp_accessed"] is False
    assert [row["articles"] for row in checks["splits"]] == [2, 1, 1]
    # Evaluation-only words would learn merges if those repeated texts leaked into fitting.
    assert "supersecretvalidationsentinel" not in tokenizer.get_vocab()
    assert "supersecrettestsentinel" not in tokenizer.get_vocab()
    assert (output / "statistics.csv").is_file()
    assert check(root, output)["tokenizer_sha256"] == manifest["tokenizer_sha256"]


def test_all_probe_round_trips_and_literal_control_strings(bundle):
    tokenizer = load_tokenizer(bundle[2])
    for text in PROBES:
        ids = encode_text(tokenizer, text)
        assert not set(ids) & set(SPECIAL_IDS.values())
        assert tokenizer.decode(ids, skip_special_tokens=False).encode() == text.encode()
    ids = encode_text(tokenizer, "<bos>", add_bos=True, add_eos=True)
    assert ids[0] == 1 and ids[-1] == 2
    assert tokenizer.decode(ids[1:-1]) == "<bos>"


def test_loader_restores_nonserialized_special_token_policy(bundle):
    path = bundle[2]
    raw = Tokenizer.from_file(str(path / "tokenizer.json"))
    assert raw.encode_special_tokens is False
    with pytest.raises(ValueError, match="load_tokenizer"):
        encode_text(raw, "<bos>")
    loaded = load_tokenizer(path)
    assert loaded.encode_special_tokens is True
    assert loaded.decode(encode_text(loaded, "<bos>")) == "<bos>"


def test_saved_tokenizer_corruption_is_rejected(bundle):
    path = bundle[2] / "tokenizer.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="Changed tokenizer artifact"):
        load_tokenizer(bundle[2])


def test_existing_bundle_is_never_overwritten(bundle):
    root, config, output = bundle
    before = sha256_file(output / "tokenizer.json")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        train(root, output, config)
    assert sha256_file(output / "tokenizer.json") == before


def test_training_is_repeatable_for_identical_inputs(bundle, tmp_path):
    root, config, output = bundle
    second = tmp_path / "second"
    train(root, second, config)
    assert sha256_file(second / "tokenizer.json") == sha256_file(output / "tokenizer.json")


@pytest.mark.parametrize("key,value", [("vocab_size", 259), ("min_frequency", 0),
                                      ("batch_size", 0), ("normalization", "lowercase"),
                                      ("add_prefix_space", True), ("parallelism", True)])
def test_invalid_or_changed_text_contract_is_rejected(key, value):
    config = read_json(DEFAULT_CONFIG)
    config[key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_cannot_silently_pad_an_underfilled_vocabulary():
    config = read_json(DEFAULT_CONFIG)
    config["vocab_size"] = 300
    tokenizer, trainer = build_tokenizer(config)
    tokenizer.train_from_iterator(["x"] * 3, trainer=trainer)
    with pytest.raises(ValueError, match="exactly 300"):
        verify_structure(tokenizer, config)


def test_stale_stage1_validation_is_rejected(fixture_data):
    root, config, output = fixture_data
    write_json(root / "validation.json", {"status": "PASS", "manifest_sha256": "stale"})
    with pytest.raises(ValueError, match="stale"):
        train(root, output, config)
    assert not output.exists()


def test_article_iterator_rejects_eval_records_in_training(fixture_data):
    root, _, _ = fixture_data
    path = root / "processed/articles/validation.jsonl"
    with pytest.raises(ValueError, match="Wrong split/order"):
        list(article_batches(path, "train", 1, 1))


def test_modified_training_input_is_rejected(fixture_data):
    root, config, output = fixture_data
    path = root / "processed/articles/train.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Changed train article file"):
        train(root, output, config)
    assert not output.exists()


def test_identical_data_reprepared_elsewhere_can_be_checked(bundle, tmp_path):
    root, _, output = bundle
    relocated = tmp_path / "friend's dataset"
    shutil.copytree(root, relocated)
    manifest = read_json(relocated / "manifest.json")
    manifest.update(data_root=str(relocated), prepared_at="different preparation time")
    write_json(relocated / "manifest.json", manifest)
    write_json(relocated / "validation.json", {
        "status": "PASS", "manifest_sha256": sha256_file(relocated / "manifest.json"),
    })
    assert check(relocated, output)["status"] == "PASS"


def test_artifact_manifest_cannot_redirect_a_named_file(bundle):
    output = bundle[2]
    manifest = read_json(output / "manifest.json")
    manifest["files"]["config.json"] = manifest["files"]["vocab.json"]
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="artifact path"):
        load_tokenizer(output)


@pytest.mark.parametrize("field,value", [("vocab_size", 301), ("special_token_ids", {}),
                                        ("config", {})])
def test_conflicting_saved_metadata_is_rejected(bundle, field, value):
    output = bundle[2]
    manifest = read_json(output / "manifest.json")
    manifest[field] = value
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="metadata"):
        load_tokenizer(output)


@pytest.mark.parametrize("failure", ["missing", "failed", "stale_statistics"])
def test_loader_requires_a_completed_check(bundle, failure):
    root, _, output = bundle
    if failure == "missing":
        (output / "checks.json").unlink()
    elif failure == "failed":
        checks = read_json(output / "checks.json")
        checks["status"] = "FAIL"
        write_json(output / "checks.json", checks)
    else:
        with (output / "statistics.csv").open("a") as stream:
            stream.write("\n")
    with pytest.raises(ValueError, match="verification"):
        load_tokenizer(output)
    # The checking command can establish the gate from a trained, unchecked bundle.
    assert check(root, output)["status"] == "PASS"
    assert load_tokenizer(output).get_vocab_size() == 300


def test_standalone_vocabulary_must_match_tokenizer_json(bundle):
    output = bundle[2]
    path = output / "vocab.json"
    vocabulary = read_json(path)
    first, second = [token for token, index in vocabulary.items() if index in (4, 5)]
    vocabulary[first], vocabulary[second] = vocabulary[second], vocabulary[first]
    write_json(path, vocabulary)
    manifest = read_json(output / "manifest.json")
    manifest["files"]["vocab.json"] = file_record(path, output)
    write_json(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="disagree"):
        load_tokenizer(output)


def test_evaluation_text_does_not_change_learned_vocabulary(bundle, tmp_path):
    root, config, output = bundle
    manifest = read_json(root / "manifest.json")
    for split in ["validation", "test"]:
        text = f"replacement evaluation text for {split} é 😀 " * 100
        path = root / "processed/articles" / f"{split}.jsonl"
        row = {"article_id": f"{split}:000000", "split": split, "text": text,
               "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
        path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest["wikitext_splits"][split].update(
            text_bytes=len(text.encode()), file=file_record(path, root),
        )
    write_json(root / "manifest.json", manifest)
    write_json(root / "validation.json", {
        "status": "PASS", "manifest_sha256": sha256_file(root / "manifest.json"),
    })
    second = tmp_path / "changed-evaluation"
    train(root, second, config)
    assert sha256_file(second / "tokenizer.json") == sha256_file(output / "tokenizer.json")
    # Rechecking an existing bundle still requires its original evaluation inputs.
    with pytest.raises(ValueError, match="split provenance"):
        check(root, output)


def test_training_is_repeatable_across_processes(bundle, tmp_path):
    root, config, output = bundle
    second = tmp_path / "fresh process"
    environment = dict(os.environ, PYTHONHASHSEED="987", TOKENIZERS_PARALLELISM="true")
    subprocess.run([
        sys.executable, "-m", "tokenizer.train", "--data-root", str(root),
        "--config", str(config), "--output", str(second),
    ], env=environment, check=True, capture_output=True, text=True)
    assert sha256_file(second / "tokenizer.json") == sha256_file(output / "tokenizer.json")
