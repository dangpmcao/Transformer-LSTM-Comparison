"""Train the shared BPE on verified WikiText training articles only.

From the repository root: python -m tokenizer.train
Checks and split-wise compression statistics run after vocabulary learning ends.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

from dataset.common import DEFAULT_DATA_ROOT, file_record, read_json, sha256_file, utc_now, write_json
from .core import (
    DEFAULT_CONFIG, DEFAULT_OUTPUT, PROBES, SPECIAL_IDS, article_batches,
    build_tokenizer, data_identity, encode_text, validate_config, verified_article_path,
    verified_data_manifest, verify_structure,
)


def train(data_root: Path, output: Path, config_path: Path = DEFAULT_CONFIG) -> None:
    data_root, output = data_root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite tokenizer artifacts: {output}; choose --output or run tokenizer.check")
    config = read_json(config_path)
    validate_config(config)
    manifest = verified_data_manifest(data_root)
    data_manifest_sha256 = sha256_file(data_root / "manifest.json")
    source_path = verified_article_path(data_root, manifest, "train")
    expected = manifest["wikitext_splits"]["train"]
    # Pin the processing mode; no sampling, shuffling, RNG seed, or BPE dropout.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = "1"
    tokenizer, trainer = build_tokenizer(config)
    print(f"Learning {config['vocab_size']:,} IDs from {expected['articles']:,} training articles (CPU).", flush=True)
    tokenizer.train_from_iterator(
        article_batches(source_path, "train", expected["articles"], config["batch_size"]),
        trainer=trainer, length=expected["articles"],
    )
    verify_structure(tokenizer, config)
    if sha256_file(source_path) != expected["file"]["sha256"]:
        raise ValueError("Training source changed during tokenizer fitting")
    if sha256_file(data_root / "manifest.json") != data_manifest_sha256:
        raise ValueError("Data manifest changed during tokenizer fitting")
    # Construct in-memory expectations BEFORE serialization to test real reload consistency.
    probes = [{"text": text, "ids": encode_text(tokenizer, text)} for text in PROBES]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "artifacts"
        staging.mkdir()
        tokenizer.save(str(staging / "tokenizer.json"))
        tokenizer.model.save(str(staging))  # vocab.json and merges.txt
        write_json(staging / "config.json", config)
        write_json(staging / "special_tokens.json", SPECIAL_IDS)
        names = ["tokenizer.json", "vocab.json", "merges.txt", "config.json", "special_tokens.json"]
        write_json(staging / "manifest.json", {
            "schema_version": 1, "status": "trained", "trained_at": utc_now(),
            "vocab_size": config["vocab_size"], "special_token_ids": SPECIAL_IDS,
            "tokenizer_sha256": sha256_file(staging / "tokenizer.json"),
            "files": {name: file_record(staging / name, staging) for name in names},
            "training_input": expected["file"], "training_articles": expected["articles"],
            "data_manifest_sha256": data_manifest_sha256,
            "data_identity": data_identity(manifest),
            "source": manifest["sources"]["wikitext"], "config": config,
            "fitting_policy": "All train articles in persisted order; no validation, test, or BLiMP fitting.",
            "text_policy": "Raw article strings, no normalizer, prefix insertion, or automatic control IDs.",
            "loading_policy": "Use tokenizer.core.load_tokenizer; it restores encode_special_tokens=True.",
            "probe_encodings_before_save": probes,
            "code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                            for name in ["train.py", "core.py", "check.py"]},
        })
        print("Vocabulary frozen. Checking saved artifacts and collecting compression statistics...", flush=True)
        from .check import check

        check(data_root, staging)
        # Publish only the complete, verified bundle. Existing bundles are never replaced.
        if output.exists():
            raise FileExistsError(f"Output appeared while training: {output}")
        staging.rename(output)
    print(f"Tokenizer ready: {output}", flush=True)
    print(f"Tokenizer SHA-256: {sha256_file(output / 'tokenizer.json')}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    train(args.data_root, args.output, args.config)


if __name__ == "__main__":
    main()
