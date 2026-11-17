"""Train the KenLM 5-gram baseline on WikiText-103 train token IDs.

Steps: export one line per training article (shared BPE IDs), estimate an interpolated
modified Kneser-Ney 5-gram with lmplz (no pruning), then convert it to a binary trie.
Only the train split is read; validation/test/BLiMP are never seen here.

Run on a shared machine with low priority, for example:
    nice -n 19 ionice -c3 python -m ngram.train --memory 16G
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from dataset.common import DEFAULT_DATA_ROOT, file_record, read_json, sha256_file, utc_now, write_json
from dataset.encode import load_cache, split_arrays
from .lm import DEFAULT_KENLM_BIN, DEFAULT_MODEL_ROOT, ORDER, iter_article_lines

DEFAULT_LABEL = "kenlm_5gram_train"


def export_corpus(token_root: Path, path: Path) -> dict:
    """Write train articles as KenLM text; returns line/word counts for the manifest."""
    cache = load_cache(token_root)
    tokens, offsets = split_arrays(token_root, "train")
    lines = words = 0
    with path.open("w", encoding="ascii", newline="\n") as stream:
        for line in iter_article_lines(tokens, offsets):
            stream.write(line + "\n")
            lines += 1
            words += len(line.split()) if line else 0
    expected = cache["splits"]["train"]
    if lines != expected["articles"] or words != expected["content_tokens"]:
        raise ValueError("Exported corpus does not match the token-cache counts")
    return {"articles": lines, "content_tokens": words, "eos_tokens_as_end_markers": lines}


def run_logged(command: list[str], log: Path) -> float:
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("$ " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return round(time.monotonic() - started, 1)


def kenlm_commit(kenlm_bin: Path) -> str | None:
    source = kenlm_bin.resolve().parents[1]
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() or None


def train(token_root: Path, output: Path, *, kenlm_bin: Path, memory: str, temp_root: Path | None,
          keep_corpus: bool) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite n-gram model: {output}")
    for tool in ("lmplz", "build_binary"):
        if not (kenlm_bin / tool).is_file():
            raise FileNotFoundError(f"Missing KenLM binary: {kenlm_bin / tool}")
    token_root = Path(token_root).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        sort_dir = Path(tempfile.mkdtemp(prefix="lmplz-", dir=temp_root or temporary))
        corpus, arpa, binary = staging / "train.txt", staging / "model.arpa", staging / "model.bin"

        started = time.monotonic()
        counts = export_corpus(token_root, corpus)
        timings = {"export_seconds": round(time.monotonic() - started, 1)}
        lmplz = [str(kenlm_bin / "lmplz"), "-o", str(ORDER), "-S", memory, "-T", str(sort_dir),
                 "--text", str(corpus), "--arpa", str(arpa)]
        timings["lmplz_seconds"] = run_logged(lmplz, staging / "lmplz.log")
        build = [str(kenlm_bin / "build_binary"), "trie", str(arpa), str(binary)]
        timings["build_binary_seconds"] = run_logged(build, staging / "build_binary.log")
        shutil.rmtree(sort_dir, ignore_errors=True)

        corpus_record = file_record(corpus, staging)
        if not keep_corpus:
            corpus.unlink()
        manifest = {
            "schema_version": 1, "status": "PASS", "created_at": utc_now(),
            "model": {"order": ORDER, "smoothing": "interpolated_modified_kneser_ney", "pruning": None,
                      "estimator": "kenlm_lmplz", "binary_format": "trie_unquantized"},
            "training_data": {"split": "train", "unit": "one_line_per_article",
                              "symbols": "shared_bpe_token_ids", "bos": "<s>", "eos": "</s>", **counts},
            "inputs": {"token_manifest_sha256": sha256_file(token_root / "manifest.json"),
                       "tokenizer_sha256": read_json(token_root / "manifest.json")["tokenizer_sha256"]},
            "kenlm": {"commit": kenlm_commit(kenlm_bin), "lmplz_command": lmplz, "build_binary_command": build,
                      "memory": memory},
            "files": {"corpus": corpus_record | {"kept": keep_corpus},
                      "arpa": file_record(arpa, staging), "binary": file_record(binary, staging)},
            "timings": timings,
        }
        write_json(staging / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(f"Output appeared during training: {output}")
        staging.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL_ROOT / DEFAULT_LABEL)
    parser.add_argument("--kenlm-bin", type=Path, default=DEFAULT_KENLM_BIN)
    parser.add_argument("--memory", default="16G", help="lmplz -S sorting memory (keep modest on shared servers)")
    parser.add_argument("--temp", type=Path, help="Directory for lmplz sort files (default: inside output parent)")
    parser.add_argument("--keep-corpus", action="store_true", help="Keep the exported train.txt")
    args = parser.parse_args()
    manifest = train(args.tokens, args.output, kenlm_bin=args.kenlm_bin, memory=args.memory,
                     temp_root=args.temp, keep_corpus=args.keep_corpus)
    print({"output": str(args.output), **manifest["timings"]})


if __name__ == "__main__":
    main()
