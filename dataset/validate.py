"""Independently verify persisted downloads, row conservation, and BLiMP IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import islice
from pathlib import Path

from .common import DEFAULT_DATA_ROOT, SOURCE_LOCK, read_json, sha256_file, utc_now, write_json
from .download import verify_file
from .prepare import article_title, iter_parquet_rows, validate_blimp


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_all(root: Path, source_lock: Path) -> dict:
    lock = read_json(source_lock)
    manifest = read_json(root / "manifest.json")
    require(manifest["source_lock_sha256"] == sha256_file(source_lock), "Source lock changed")
    for specification in lock["files"]:
        verify_file(root / specification["path"], specification)
    checked = {}
    for split, summary in manifest["wikitext_splits"].items():
        source = iter_parquet_rows(sorted((root / "raw/wikitext103_raw").glob(f"{split}-*.parquet")))
        output = root / summary["file"]["path"]
        require(sha256_file(output) == summary["file"]["sha256"], f"Changed article file: {split}")
        metadata_path = root / summary["article_manifest"]["path"]
        require(sha256_file(metadata_path) == summary["article_manifest"]["sha256"], f"Changed article index: {split}")
        metadata = read_json(metadata_path)
        end = count = 0
        with output.open(encoding="utf-8") as stream:
            for count, line in enumerate(stream, 1):
                item = json.loads(line)
                require(item["split"] == split, f"Split mismatch: {item['article_id']}")
                require(item["article_id"] == f"{split}:{count-1:06d}", "Article IDs are not ordered/unique")
                require(item["row_start"] == end, "Missing or repeated source rows")
                rows = list(islice(source, item["row_end"] - item["row_start"]))
                require(len(rows) == item["row_end"] - item["row_start"], "Source exhausted early")
                require("".join(rows) == item["text"], f"Text was altered: {item['article_id']}")
                heading_index = item["heading_row"] - item["row_start"]
                require(0 <= heading_index < len(rows), "Heading outside article range")
                require(article_title(rows[heading_index]) == item["title"], "Incorrect title")
                require(heading_index == 0 or not rows[heading_index-1].strip(), "Header lacks preceding blank row")
                require(heading_index+1 == len(rows) or not rows[heading_index+1].strip(), "Header lacks following blank row")
                for j in range(heading_index+1, len(rows)):
                    isolated = not rows[j-1].strip() and (j+1 == len(rows) or not rows[j+1].strip())
                    require(not (isolated and article_title(rows[j])), "Missed article boundary")
                digest = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
                require(digest == item["text_sha256"], "Changed article text hash")
                require(count <= len(metadata), "Article metadata index truncated")
                require({k: v for k, v in item.items() if k != "text"} == metadata[count-1], "Article metadata mismatch")
                end = item["row_end"]
        require(next(source, None) is None, f"Unconsumed source rows: {split}")
        require(end == summary["rows"] == lock["wikitext"]["expected_rows"][split], "Row count changed")
        require(count == len(metadata) == summary["articles"], "Article count changed")
        checked[split] = {"rows": end, "articles": count, "exact_source_text": True}
    for record in [manifest["duplicates"]["file"], manifest["blimp"]["mapping"]]:
        require(sha256_file(root / record["path"]) == record["sha256"], f"Changed artifact: {record['path']}")
    blimp = validate_blimp(root, lock["blimp"], save_mapping=False)
    require(blimp["groups"] == manifest["blimp"]["groups"], "BLiMP grouping changed")
    require(blimp["identical_candidate_pairs"] == manifest["blimp"]["identical_candidate_pairs"], "BLiMP anomalies changed")
    result = {
        "status": "PASS", "validated_at": utc_now(), "data_root": str(root.resolve()),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "verified_download_files": len(lock["files"]), "wikitext": checked,
        "blimp": {"paradigms": blimp["paradigms"], "pairs": blimp["pairs"], "groups": len(blimp["groups"])},
        "scope": "Data integrity/preprocessing only; no model or tokenizer training and no BLiMP scoring",
    }
    write_json(root / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    args = parser.parse_args()
    print(json.dumps(validate_all(args.data_root.resolve(), args.source_lock), indent=2))


if __name__ == "__main__":
    main()
