"""Download revision-pinned raw WikiText-103 and evaluation-only BLiMP.

Run from the checkout: python -m dataset.download
Default dataset storage is ../dataset, outside the code checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .common import (
    DEFAULT_DATA_ROOT, SOURCE_LOCK, contained_path, file_record,
    read_json, sha256_file, utc_now, write_json,
)


def make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "COMP4221-Stage1/1.0"
    return session


def verify_file(path: Path, specification: dict[str, Any]) -> None:
    size = path.stat().st_size
    if size != specification["size"]:
        raise ValueError(f"Size mismatch for {path}: {size} != {specification['size']}")
    if "sha256" in specification:
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {path}")
    if "git_blob_sha1" in specification:
        digest = hashlib.sha1(f"blob {size}\0".encode(), usedforsecurity=False)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != specification["git_blob_sha1"]:
            raise ValueError(f"Git blob hash mismatch for {path}")


def download_one(root: Path, specification: dict[str, Any]) -> dict[str, Any]:
    destination = contained_path(root, specification["path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        # Never overwrite a conflicting retained artifact.
        verify_file(destination, specification)
    else:
        partial = destination.with_suffix(destination.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > specification["size"]:
            raise ValueError(f"Oversized retained partial file: {partial}")
        if offset != specification["size"]:
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with make_session() as session:
                with session.get(
                    specification["url"], headers=headers, stream=True, timeout=(30, 120)
                ) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise ValueError(f"Unexpected partial response: {content_range}")
                        mode = "ab"
                    elif response.status_code == 200:
                        mode = "wb"  # Server ignored Range; restart only the partial file.
                    else:
                        raise ValueError(f"Unexpected download status {response.status_code}")
                    with partial.open(mode) as stream:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                stream.write(chunk)
        verify_file(partial, specification)
        os.replace(partial, destination)
    return {**file_record(destination, root), "source": specification["source"],
            "url": specification["url"]}


def download_all(root: Path, source_lock: Path, workers: int) -> dict[str, Any]:
    lock = read_json(source_lock)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, root, item): item for item in lock["files"]}
        for future in as_completed(futures):
            records.append(future.result())
            print(f"Verified {futures[future]['path']}", flush=True)
    manifest = {
        "schema_version": 1, "status": "verified_downloads", "verified_at": utc_now(),
        "data_root": str(root.resolve()), "source_lock_sha256": sha256_file(source_lock),
        "sources": {name: lock[name] for name in ["wikitext", "blimp"]},
        "files": sorted(records, key=lambda item: item["path"]),
        "blimp_use": "evaluation_only; excluded from tokenizer and model training",
    }
    write_json(root / "downloads.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--source-lock", type=Path, default=SOURCE_LOCK)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = download_all(args.data_root.resolve(), args.source_lock, args.workers)
    print(json.dumps({"status": manifest["status"], "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
