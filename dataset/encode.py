"""Persist frozen-tokenizer IDs and article offsets outside the source checkout."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tokenizer.core import (
    DEFAULT_OUTPUT, SPECIAL_IDS, article_batches, data_identity, load_tokenizer,
    verified_article_path, verified_data_manifest,
)
from .common import (
    DEFAULT_DATA_ROOT, contained_path, file_record, read_json, sha256_file,
    utc_now, write_json,
)

SPLITS = ("train", "validation", "test")


def load_cache(root: Path) -> dict[str, Any]:
    """Verify every cache file before exposing memory-mapped arrays to a trainer."""
    root = Path(root).resolve()
    manifest = read_json(root / "manifest.json")
    if (manifest.get("schema_version") != 1 or manifest.get("status") != "PASS"
            or manifest.get("special_token_ids") != SPECIAL_IDS
            or manifest.get("token_dtype") != "<u2"
            or manifest.get("offset_dtype") != "<u8"
            or set(manifest.get("splits", {})) != set(SPLITS)):
        raise ValueError("Invalid token cache manifest")
    for split, summary in manifest["splits"].items():
        for key, suffix in (("tokens", "tokens"), ("offsets", "offsets")):
            record = summary[key]
            if record["path"] != f"{split}.{suffix}.npy":
                raise ValueError("Unexpected token-cache path")
            path = contained_path(root, record["path"])
            if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
                raise ValueError(f"Changed token cache: {path.name}")
        tokens, offsets = split_arrays(root, split)
        if (tokens.dtype.str != "<u2" or offsets.dtype.str != "<u8"
                or tokens.ndim != 1 or offsets.shape != (summary["articles"] + 1,)
                or offsets[0] != 0 or offsets[-1] != len(tokens)
                or np.any(offsets[1:] <= offsets[:-1])
                or len(tokens) != summary["prediction_tokens"]
                or summary["prediction_tokens"] != summary["content_tokens"] + summary["articles"]):
            raise ValueError(f"Invalid token/offset structure: {split}")
        if np.any(tokens[offsets[1:].astype(np.int64) - 1] != SPECIAL_IDS["<eos>"]):
            raise ValueError(f"Missing article EOS: {split}")
        eos_count = 0
        for start in range(0, len(tokens), 1_000_000):
            chunk = tokens[start:start + 1_000_000]
            if np.any(chunk >= manifest["vocab_size"]) or np.isin(chunk, [0, 1, 3]).any():
                raise ValueError(f"Control/unknown/out-of-range cached ID: {split}")
            eos_count += int(np.count_nonzero(chunk == SPECIAL_IDS["<eos>"]))
        if eos_count != summary["articles"]:
            raise ValueError(f"Expected exactly one EOS per article: {split}")
    return manifest


def split_arrays(root: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return tuple(np.load(Path(root) / f"{split}.{name}.npy", mmap_mode="r", allow_pickle=False)
                 for name in ("tokens", "offsets"))


def encode(data_root: Path, artifacts: Path = DEFAULT_OUTPUT, output: Path | None = None) -> dict:
    """Encode all splits, appending one EOS per article and never storing BOS/PAD."""
    data_root, artifacts = Path(data_root).resolve(), Path(artifacts).resolve()
    output = Path(output).resolve() if output else data_root / "processed/tokens"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite token cache: {output}; use --check or another --output")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = load_tokenizer(artifacts)
    saved = read_json(artifacts / "manifest.json")
    checks = read_json(artifacts / "checks.json")
    data = verified_data_manifest(data_root)
    if saved["data_identity"] != data_identity(data):
        raise ValueError("Dataset identity differs from the frozen tokenizer")
    if not 4 < saved["vocab_size"] <= 65536:
        raise ValueError("The uint16 cache requires a vocabulary of at most 65,536 IDs")
    expected = {row["split"]: row for row in checks["splits"]}
    source_paths = {split: verified_article_path(data_root, data, split) for split in SPLITS}
    source_manifest_hash = sha256_file(data_root / "manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "cache"
        staging.mkdir()
        summaries = {}
        for split in SPLITS:
            summary, reference = data["wikitext_splits"][split], expected[split]
            if reference["articles"] != summary["articles"]:
                raise ValueError("Tokenizer check/article count mismatch")
            total = reference["bpe_tokens"] + summary["articles"]
            tokens = np.lib.format.open_memmap(
                staging / f"{split}.tokens.npy", mode="w+", dtype="<u2", shape=(total,),
            )
            offsets = np.zeros(summary["articles"] + 1, dtype="<u8")
            cursor = count = 0
            print(f"Encoding {split}: {summary['articles']:,} articles...", flush=True)
            for batch in article_batches(source_paths[split], split, summary["articles"], saved["config"]["batch_size"]):
                for encoding in tokenizer.encode_batch(batch, add_special_tokens=False):
                    ids = encoding.ids
                    if any(i < 4 or i >= saved["vocab_size"] for i in ids):
                        raise ValueError(f"Control/unknown/out-of-range content ID: {split}")
                    end = cursor + len(ids) + 1
                    if end > total:
                        raise ValueError("Encoded token count exceeds saved tokenizer statistics")
                    tokens[cursor:end - 1] = ids
                    tokens[end - 1] = SPECIAL_IDS["<eos>"]
                    count += 1
                    offsets[count] = end
                    cursor = end
                if count % 2048 == 0:
                    print(f"  {split}: {count:,} articles, {cursor:,} targets", flush=True)
            if cursor != total or count != summary["articles"]:
                raise ValueError("Encoded counts differ from the verified tokenizer statistics")
            tokens.flush()
            del tokens
            np.save(staging / f"{split}.offsets.npy", offsets, allow_pickle=False)
            if sha256_file(source_paths[split]) != summary["file"]["sha256"]:
                raise ValueError("Article source changed during encoding")
            summaries[split] = {
                "articles": count, "content_tokens": reference["bpe_tokens"],
                "prediction_tokens": total,
                **{name: file_record(staging / f"{split}.{name}.npy", staging)
                   for name in ("tokens", "offsets")},
            }
        if sha256_file(data_root / "manifest.json") != source_manifest_hash:
            raise ValueError("Data manifest changed during encoding")
        # Recheck the source bundle after the long encoding operation.
        load_tokenizer(artifacts)
        if sha256_file(artifacts / "manifest.json") != checks["manifest_sha256"]:
            raise ValueError("Tokenizer changed during encoding")
        write_json(staging / "manifest.json", {
            "schema_version": 1, "status": "PASS", "created_at": utc_now(),
            "token_dtype": "<u2", "offset_dtype": "<u8",
            "vocab_size": saved["vocab_size"], "special_token_ids": SPECIAL_IDS,
            "tokenizer_sha256": saved["tokenizer_sha256"],
            "tokenizer_manifest_sha256": checks["manifest_sha256"],
            "data_manifest_sha256": source_manifest_hash, "data_identity": data_identity(data),
            "encoding_policy": "Content followed by exactly one EOS per article; no BOS or PAD.",
            "article_ids": "{split}:{zero_based_article_index:06d}; offsets are half-open",
            "code_sha256": sha256_file(Path(__file__)), "splits": summaries,
        })
        result = load_cache(staging)
        if output.exists():
            raise FileExistsError(f"Output appeared during encoding: {output}")
        staging.rename(output)
    print(f"Token cache ready: {output}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true", help="Verify existing cache instead of encoding")
    args = parser.parse_args()
    if args.check:
        result = load_cache(args.output or args.data_root / "processed/tokens")
    else:
        result = encode(args.data_root, args.artifacts, args.output)
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
