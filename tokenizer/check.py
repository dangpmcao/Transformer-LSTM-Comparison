"""Check a frozen tokenizer and report compression without changing its vocabulary."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from dataset.common import DEFAULT_DATA_ROOT, read_json, sha256_file, utc_now, write_json
from .core import (
    DEFAULT_OUTPUT, PROBES, SPECIAL_IDS, article_batches, data_identity, encode_text, load_tokenizer,
    verified_article_path, verified_data_manifest,
)


def check(data_root: Path, artifacts: Path) -> dict:
    data_root, artifacts = data_root.resolve(), artifacts.resolve()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = load_tokenizer(artifacts, require_checks=False)
    saved = read_json(artifacts / "manifest.json")
    data = verified_data_manifest(data_root)
    # Preparation timestamps and absolute dataset roots differ between machines.
    # Require identical source revision, split bytes, and counts instead.
    if saved["data_identity"] != data_identity(data):
        raise ValueError("WikiText source/split provenance differs from the frozen bundle")
    if saved["training_input"] != data["wikitext_splits"]["train"]["file"]:
        raise ValueError("Tokenizer training-input provenance differs from the verified training split")
    for text in PROBES:
        ids = encode_text(tokenizer, text)
        if tokenizer.decode(ids, skip_special_tokens=False).encode("utf-8") != text.encode("utf-8"):
            raise ValueError(f"Round-trip failure for probe {text!r}")
    for probe in saved["probe_encodings_before_save"]:
        if encode_text(tokenizer, probe["text"]) != probe["ids"]:
            raise ValueError("Save/reload changed an encoding")
    content = encode_text(tokenizer, "<bos> Café 😀\n")
    bounded = encode_text(tokenizer, "<bos> Café 😀\n", add_bos=True, add_eos=True)
    if bounded != [SPECIAL_IDS["<bos>"], *content, SPECIAL_IDS["<eos>"]]:
        raise ValueError("Incorrect explicit BOS/EOS insertion")
    # Batch behavior must agree exactly with the single-text API.
    if [e.ids for e in tokenizer.encode_batch(list(PROBES), add_special_tokens=False)] != [encode_text(tokenizer, text) for text in PROBES]:
        raise ValueError("Batched and single-text encodings disagree")
    rows = []
    for split in ["train", "validation", "test"]:
        path = verified_article_path(data_root, data, split)
        summary = data["wikitext_splits"][split]
        lengths = []
        nbytes = words = tokens = 0
        print(f"Checking every {split} article...", flush=True)
        for batch in article_batches(path, split, summary["articles"], saved["config"]["batch_size"]):
            encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
            for text, encoding in zip(batch, encodings, strict=True):
                ids = encoding.ids
                if any(token_id < 4 or token_id >= saved["vocab_size"] for token_id in ids):
                    raise ValueError(f"Unexpected control/unknown/out-of-range ID in {split} text")
                if tokenizer.decode(ids, skip_special_tokens=False).encode("utf-8") != text.encode("utf-8"):
                    raise ValueError(f"Round-trip failed in {split}")
                nbytes += len(text.encode("utf-8"))
                words += len(text.split())
                tokens += len(ids)
                lengths.append(len(ids))
        if nbytes != summary["text_bytes"]:
            raise ValueError(f"Encoded byte count differs from Stage 1 for {split}")
        rows.append({
            "split": split, "articles": len(lengths), "bytes": nbytes,
            "whitespace_words": words, "bpe_tokens": tokens,
            "tokens_per_word": tokens / words if words else 0.0,
            "bytes_per_token": nbytes / tokens if tokens else 0.0,
            **{f"article_tokens_p{p}": float(np.percentile(lengths, p))
               for p in [0, 25, 50, 75, 95, 99, 100]},
        })
        if sha256_file(path) != summary["file"]["sha256"]:
            raise ValueError(f"{split} source changed during verification")
    # These are descriptive content-token statistics; BOS/EOS/PAD are excluded.
    temporary = artifacts / "statistics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(artifacts / "statistics.csv")
    result = {
        "status": "PASS", "checked_at": utc_now(),
        "tokenizer_sha256": sha256_file(artifacts / "tokenizer.json"),
        "manifest_sha256": sha256_file(artifacts / "manifest.json"),
        "data_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "statistics_sha256": sha256_file(artifacts / "statistics.csv"),
        "vocab_size": tokenizer.get_vocab_size(), "special_token_ids": SPECIAL_IDS,
        "probes": len(PROBES), "every_article_round_trip": True,
        "reserved_ids_in_content": 0, "splits": rows,
        "statistics_policy": "All WikiText splits, after freezing; no special IDs counted, whitespace-defined words.",
        "fitting_split": "train", "blimp_accessed": False,
    }
    write_json(artifacts / "checks.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = check(args.data_root, args.artifacts)
    print(json.dumps({"status": result["status"], "tokenizer_sha256": result["tokenizer_sha256"]}))


if __name__ == "__main__":
    main()
