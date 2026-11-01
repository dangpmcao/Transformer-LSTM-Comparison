"""Reconstruct articles without altering raw text; audit evaluation-only BLiMP.

WikiText rows are concatenated exactly, with no inserted/removed whitespace.
Top-level headings start articles; subsection headings remain inside articles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .common import (
    DEFAULT_DATA_ROOT, SOURCE_LOCK, file_record, read_json,
    sha256_file, utc_now, write_json,
)
from .download import verify_file

ARTICLE_HEADING = re.compile(r"^=[ \t]+([^=\s].*?)[ \t]+=$")
EXPECTED_GROUPS = {
    "anaphor_agreement", "argument_structure", "binding", "control_raising",
    "determiner_noun_agreement", "ellipsis", "filler_gap_dependency",
    "irregular_forms", "island_effects", "npi_licensing", "quantifiers",
    "subject_verb_agreement",
}
# Published appendix groups both animate-subject paradigms under ARGUMENT STRUCTURE:
# https://aclanthology.org/2020.tacl-1.25.pdf (printed page 389).
# Preserve the original 13 raw labels, expose a separate 12-group paper mapping.
GROUP_ALIASES = {"s-selection": "argument_structure"}


def article_title(text: str) -> str | None:
    match = ARTICLE_HEADING.fullmatch(text.strip())
    return match.group(1) if match else None


def iter_parquet_rows(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=4096, columns=["text"]):
            for text in batch.column(0).to_pylist():
                if not isinstance(text, str):
                    raise ValueError(f"Non-string WikiText row in {path}")
                yield text


def iter_articles(rows: Iterable[str], split: str) -> Iterator[dict[str, Any]]:
    """Partition all rows, attaching leading blank rows to the first article.

    Row ranges are half-open and refer to the concatenated official split.
    Unknown nonblank preambles fail visibly rather than losing source text.
    """
    pieces: list[str] = []
    title: str | None = None
    start = heading_row = article_index = 0
    end = 0

    def record(row_end: int) -> dict[str, Any]:
        text = "".join(pieces)
        return {
            "article_id": f"{split}:{article_index:06d}", "split": split,
            "title": title, "row_start": start, "row_end": row_end,
            "heading_row": heading_row, "text": text,
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }

    source = iter(rows)
    text = next(source, None)
    if text is None:
        return
    previous_blank = True
    for index, following in enumerate(chain(source, [None])):
        end = index + 1
        # Raw WikiText also contains equation/table fragments formatted '= x ='.
        # Real article headers are isolated by blank rows on both sides.
        isolated = previous_blank and (following is None or not following.strip())
        new_title = article_title(text) if isolated else None
        if new_title is not None:
            if title is not None:
                yield record(index)
                article_index += 1
                start = index
                pieces = []
            title = new_title
            heading_row = index
        elif title is None and text.strip():
            raise ValueError(f"Nonblank text before first article: {split} row {index}")
        pieces.append(text)
        previous_blank = not text.strip()
        text = following
    if title is not None:
        yield record(end)
    elif pieces:
        raise ValueError(f"No article heading found in nonempty split {split}")


def prepare_split(root: Path, split: str, expected_rows: int) -> dict[str, Any]:
    paths = sorted((root / "raw/wikitext103_raw").glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet shards for {split}")
    output = root / "processed/articles" / f"{split}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.tmp")
    source_digest = hashlib.sha256()
    reconstructed_digest = hashlib.sha256()
    row_count = nonempty_rows = unk_count = 0

    def rows() -> Iterator[str]:
        nonlocal row_count, nonempty_rows, unk_count
        for text in iter_parquet_rows(paths):
            row_count += 1
            nonempty_rows += bool(text.strip())
            unk_count += text.count("<unk>")
            source_digest.update(text.encode("utf-8"))
            yield text

    articles = []
    byte_lengths, char_lengths = [], []
    samples = []
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for article in iter_articles(rows(), split):
            stream.write(json.dumps(article, ensure_ascii=False) + "\n")
            encoded = article["text"].encode("utf-8")
            reconstructed_digest.update(encoded)
            byte_lengths.append(len(encoded))
            char_lengths.append(len(article["text"]))
            if len(samples) < 2:
                samples.append({
                    "article_id": article["article_id"], "title": article["title"],
                    "preview": article["text"][:300],
                })
            articles.append({key: value for key, value in article.items() if key != "text"})
    if row_count != expected_rows:
        raise ValueError(f"Row count mismatch for {split}: {row_count} != {expected_rows}")
    if source_digest.digest() != reconstructed_digest.digest():
        raise ValueError(f"Reconstructed text differs from source rows in {split}")
    # Exact source-row conservation is independently checked in validate.py.
    os.replace(temporary, output)
    article_manifest = root / "processed/articles" / f"{split}.manifest.json"
    write_json(article_manifest, articles)
    import numpy as np

    return {
        "rows": row_count, "nonempty_rows": nonempty_rows, "articles": len(articles),
        "literal_unk_count": unk_count, "text_bytes": sum(byte_lengths),
        "text_characters": sum(char_lengths), "concatenated_text_sha256": source_digest.hexdigest(),
        "article_byte_percentiles": {
            str(p): float(np.percentile(byte_lengths, p)) for p in [0, 25, 50, 75, 95, 99, 100]
        },
        "file": file_record(output, root),
        "article_manifest": file_record(article_manifest, root), "inspection_samples": samples,
    }


def duplicate_audit(root: Path) -> dict[str, Any]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split in ["train", "validation", "test"]:
        for item in read_json(root / "processed/articles" / f"{split}.manifest.json"):
            groups[item["text_sha256"]].append({
                "article_id": item["article_id"], "split": split, "title": item["title"],
            })
    duplicates = [items for items in groups.values() if len(items) > 1]
    cross_split = [items for items in duplicates if len({x["split"] for x in items}) > 1]
    return {
        "method": "SHA-256 of exact article UTF-8 text, including headings and whitespace",
        "duplicate_groups": duplicates, "cross_split_duplicate_groups": cross_split,
        "within_split_group_counts": {
            split: sum(sum(item["split"] == split for item in group) > 1 for group in duplicates)
            for split in ["train", "validation", "test"]
        },
        "action": "Report only; preserve every official split and article",
    }


def validate_blimp(root: Path, expected: dict[str, Any], *, save_mapping: bool = True) -> dict[str, Any]:
    paths = sorted((root / "raw/blimp/data").glob("*.jsonl"))
    if len(paths) != expected["expected_paradigms"]:
        raise ValueError(f"Expected {expected['expected_paradigms']} BLiMP paradigms, got {len(paths)}")
    groups: Counter[str] = Counter()
    raw_groups: Counter[str] = Counter()
    mapping, samples, identical_candidates = [], [], []
    total = 0
    required = {"sentence_good", "sentence_bad", "field", "linguistics_term", "UID", "pairID"}
    for path in paths:
        ids = set()
        group = field = None
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                item = json.loads(line)
                if not required.issubset(item):
                    raise ValueError(f"Missing BLiMP fields: {path}:{line_number}")
                if item["UID"] != path.stem:
                    raise ValueError(f"Paradigm UID mismatch: {path}:{line_number}")
                pair_id = str(item["pairID"])
                if pair_id in ids:
                    raise ValueError(f"Duplicate pair ID: {path}:{pair_id}")
                ids.add(pair_id)
                for key in ["sentence_good", "sentence_bad", "field", "linguistics_term"]:
                    if not isinstance(item[key], str) or not item[key].strip():
                        raise ValueError(f"Empty/invalid {key}: {path}:{line_number}")
                if item["sentence_good"] == item["sentence_bad"]:
                    # Preserve upstream defects and IDs; never silently filter benchmark pairs.
                    identical_candidates.append({"UID": item["UID"], "pairID": pair_id})
                if group is not None and (group, field) != (item["linguistics_term"], item["field"]):
                    raise ValueError(f"Inconsistent BLiMP grouping: {path}:{line_number}")
                group, field = item["linguistics_term"], item["field"]
                if len(samples) < 2:
                    samples.append({k: item[k] for k in sorted(required)})
        expected_count = expected["pairs_per_paradigm"]
        if ids != {str(i) for i in range(expected_count)}:
            raise ValueError(f"Expected pair IDs 0..{expected_count - 1}: {path}")
        paper_group = GROUP_ALIASES.get(group, group)
        groups[paper_group] += len(ids)
        raw_groups[group] += len(ids)
        total += len(ids)
        mapping.append({"paradigm": path.stem, "group": paper_group,
                        "source_linguistics_term": group, "field": field, "pairs": len(ids)})
    if set(groups) != EXPECTED_GROUPS or len(groups) != expected["expected_groups"]:
        raise ValueError(f"Unexpected BLiMP groups: {sorted(groups)}")
    mapping_path = root / "processed/blimp/paradigm_groups.json"
    if save_mapping:
        write_json(mapping_path, mapping)
    return {
        "paradigms": len(paths), "pairs": total, "groups": dict(sorted(groups.items())),
        "raw_linguistics_terms": dict(sorted(raw_groups.items())),
        "group_aliases": GROUP_ALIASES,
        "group_mapping_source": "https://aclanthology.org/2020.tacl-1.25.pdf#page=13",
        "pair_id_scope": "(UID, pairID)", "use": "evaluation_only",
        "mapping": file_record(mapping_path, root), "inspection_samples": samples,
        "identical_candidate_pairs": identical_candidates,
        "anomaly_policy": "Preserve all official pairs; flag identical candidates for explicit Stage 6 tie/sensitivity reporting.",
    }


def license_records(root: Path) -> dict[str, Any]:
    wt = (root / "raw/wikitext103_raw/README.md").read_text(encoding="utf-8")
    bl = (root / "raw/blimp/README.md").read_text(encoding="utf-8")
    if "cc-by-sa-3.0" not in wt or "gfdl" not in wt:
        raise ValueError("WikiText source license metadata changed")
    if "creativecommons.org/licenses/by/4.0/" not in bl:
        raise ValueError("BLiMP source no longer supplies expected CC-BY-4.0 statement")
    result = {
        "wikitext": {
            "source": "Salesforce/wikitext, pinned README.md",
            "metadata_licenses": ["CC-BY-SA-3.0", "GFDL"],
            "card_body_licenses": ["CC-BY-SA-4.0"],
            "status": "source_inconsistency_recorded",
            "note": "Upstream metadata and prose disagree; retain both and verify terms before redistribution.",
            "attribution": "Stephen Merity, Caiming Xiong, James Bradbury, Richard Socher; Pointer Sentinel Mixture Models (2016).",
            "source_url": "https://huggingface.co/datasets/Salesforce/wikitext",
        },
        "blimp": {
            "license": "CC-BY-4.0", "status": "confirmed_in_pinned_source_README",
            "attribution": "Alex Warstadt et al.; BLiMP: The Benchmark of Linguistic Minimal Pairs for English, TACL 8 (2020), 377–392, doi:10.1162/tacl_a_00321.",
            "source_url": "https://github.com/alexwarstadt/blimp",
        },
    }
    write_json(root / "ATTRIBUTION.json", result)
    return result


def prepare_all(root: Path, source_lock: Path) -> dict[str, Any]:
    lock = read_json(source_lock)
    downloads = read_json(root / "downloads.json")
    if downloads["source_lock_sha256"] != sha256_file(source_lock):
        raise ValueError("Source lock differs from completed download manifest")
    for specification in lock["files"]:
        verify_file(root / specification["path"], specification)
    summaries = {}
    for split, count in lock["wikitext"]["expected_rows"].items():
        summaries[split] = prepare_split(root, split, count)
        print(f"Prepared {split}: {summaries[split]['articles']} articles", flush=True)
    duplicates = duplicate_audit(root)
    write_json(root / "processed/duplicates.json", duplicates)
    result = {
        "schema_version": 1, "status": "prepared_and_verified", "prepared_at": utc_now(),
        "data_root": str(root.resolve()), "source_lock_sha256": sha256_file(source_lock),
        "sources": downloads["sources"], "download_files": downloads["files"],
        "preprocessing": {
            "article_boundary": ARTICLE_HEADING.pattern,
            "article_boundary_context": "Heading row must be preceded and followed by a blank row (or split boundary); equation/table fragments stay inside their article.",
            "text_policy": "Concatenate source rows exactly; preserve case, Unicode, formatting and headings.",
            "leading_blank_rows": "Retained as part of the first article in each split",
            "row_range": "[row_start, row_end), zero-based across source shards",
            "normalization": "none", "split_policy": "unchanged official splits",
            "code_sha256": sha256_file(Path(__file__)),
        },
        "wikitext_splits": summaries,
        "blimp": validate_blimp(root, lock["blimp"]),
        "duplicates": {"all_groups": len(duplicates["duplicate_groups"]),
                       "cross_split_groups": len(duplicates["cross_split_duplicate_groups"]),
                       "within_split_groups": duplicates["within_split_group_counts"],
                       "file": file_record(root / "processed/duplicates.json", root)},
        "licenses": license_records(root),
        "training_access": "Only processed/articles/train.jsonl is eligible for tokenizer/model fitting.",
    }
    write_json(root / "manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    args = parser.parse_args()
    result = prepare_all(args.data_root.resolve(), args.source_lock)
    print(json.dumps({"status": result["status"], "blimp_pairs": result["blimp"]["pairs"]}))


if __name__ == "__main__":
    main()
