"""Shared independent 512-target windows, deterministic ordering, and masked NLL."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from tokenizer.core import SPECIAL_IDS
from .common import DEFAULT_DATA_ROOT, contained_path, file_record, read_json, sha256_file, utc_now, write_json
from .encode import SPLITS, load_cache, split_arrays

WINDOW_TARGETS = 512


def make_index(offsets: np.ndarray) -> np.ndarray:
    """Rows are (article index, target start within article, valid targets)."""
    rows = []
    for article, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        length = int(end) - int(start)
        if length <= 0:
            raise ValueError("Every encoded article must contain at least EOS")
        for local in range(0, length, WINDOW_TARGETS):
            rows.append((article, local, min(WINDOW_TARGETS, length - local)))
    return np.asarray(rows, dtype="<i8").reshape(-1, 3)


def load_windows(root: Path, token_root: Path) -> dict:
    """Verify frozen ordering and coverage, including short/EOS-only final windows."""
    root, token_root = Path(root).resolve(), Path(token_root).resolve()
    cache = load_cache(token_root)
    manifest = read_json(root / "manifest.json")
    if (manifest.get("status") != "PASS" or manifest.get("schema_version") != 1
            or manifest.get("window_targets") != WINDOW_TARGETS
            or manifest.get("tokenizer_sha256") != cache["tokenizer_sha256"]
            or manifest.get("token_manifest_sha256") != sha256_file(token_root / "manifest.json")
            or set(manifest.get("splits", {})) != set(SPLITS)):
        raise ValueError("Window manifest differs from the token cache/512-target contract")
    for split, summary in manifest["splits"].items():
        record = summary["index"]
        if record["path"] != f"{split}.windows.npy":
            raise ValueError("Unexpected window-index path")
        path = contained_path(root, record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Changed window index: {split}")
        index = np.load(path, allow_pickle=False)
        _, offsets = split_arrays(token_root, split)
        if (index.dtype.str != "<i8" or index.shape != (summary["windows"], 3)
                or not np.array_equal(index, make_index(offsets))
                or int(index[:, 2].sum()) != cache["splits"][split]["prediction_tokens"]
                or summary["prediction_tokens"] != cache["splits"][split]["prediction_tokens"]
                or summary["full_windows"] != int(np.count_nonzero(index[:, 2] == WINDOW_TARGETS))):
            raise ValueError(f"Window index does not cover articles exactly once: {split}")
    return manifest


def build_windows(token_root: Path, output: Path | None = None) -> dict:
    token_root = Path(token_root).resolve()
    output = Path(output).resolve() if output else token_root / "windows"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite windows: {output}; use --check or another --output")
    cache = load_cache(token_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / "windows"
        staging.mkdir()
        splits = {}
        for split in SPLITS:
            _, offsets = split_arrays(token_root, split)
            index = make_index(offsets)
            np.save(staging / f"{split}.windows.npy", index, allow_pickle=False)
            splits[split] = {
                "windows": len(index), "full_windows": int(np.count_nonzero(index[:, 2] == WINDOW_TARGETS)),
                "prediction_tokens": cache["splits"][split]["prediction_tokens"],
                "index": file_record(staging / f"{split}.windows.npy", staging),
            }
        write_json(staging / "manifest.json", {
            "schema_version": 1, "status": "PASS", "created_at": utc_now(),
            "window_targets": WINDOW_TARGETS, "index_columns": ["article_index", "target_start", "valid_targets"],
            "token_manifest_sha256": sha256_file(token_root / "manifest.json"),
            "tokenizer_sha256": cache["tokenizer_sha256"],
            "ordering": "Split article order, then nonoverlapping target starts; fixed for validation/test.",
            "window_ids": "{split}:{article_index:06d}:{target_start:09d}",
            "alignment": "Each window starts with BOS and predicts its own targets; no preceding-window history.",
            "state_policy": "Reset positions and both LSTM states at every window, including within articles.",
            "code_sha256": sha256_file(Path(__file__)), "splits": splits,
        })
        result = load_windows(staging, token_root)
        if output.exists():
            raise FileExistsError(f"Output appeared during window construction: {output}")
        staging.rename(output)
    return result


class ArticleWindows(Dataset):
    """CPU tensors [time=512]; models later produce logits [batch, time, vocab]."""

    def __init__(self, token_root: Path, split: str, window_root: Path | None = None):
        self.token_root = Path(token_root).resolve()
        self.window_root = Path(window_root).resolve() if window_root else self.token_root / "windows"
        self.manifest = load_windows(self.window_root, self.token_root)
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self.tokens, self.offsets = split_arrays(self.token_root, split)
        self.index = np.load(self.window_root / f"{split}.windows.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        if not 0 <= index < len(self):
            raise IndexError(index)
        article, local, valid = map(int, self.index[index])
        start = int(self.offsets[article]) + local
        # Copy only this small window; torch must not write through a read-only memmap.
        target_ids = torch.full((WINDOW_TARGETS,), SPECIAL_IDS["<pad>"], dtype=torch.long)
        target_ids[:valid] = torch.from_numpy(np.array(self.tokens[start:start + valid], dtype=np.int64))
        input_ids = torch.full_like(target_ids, SPECIAL_IDS["<pad>"])
        input_ids[0] = SPECIAL_IDS["<bos>"]
        input_ids[1:valid] = target_ids[:valid - 1]
        positions = torch.arange(WINDOW_TARGETS, dtype=torch.long)
        mask = positions < valid
        return {
            "input_ids": input_ids, "target_ids": target_ids,
            "loss_mask": mask, "attention_mask": mask.clone(), "position_ids": positions,
            "window_id": f"{self.split}:{article:06d}:{local:09d}",
        }


class WindowOrder(Sampler[int]):
    """Paired NumPy PCG64 permutations, independent of a model's torch RNG."""

    def __init__(self, dataset: ArticleWindows, *, training: bool, seed: int = 13, epoch: int = 0):
        if type(seed) is not int or type(epoch) is not int or seed < 0 or epoch < 0:
            raise ValueError("Seed and epoch must be nonnegative")
        if training and dataset.split != "train":
            raise ValueError("Validation/test order must never be shuffled")
        self.size, self.training, self.seed, self.epoch = len(dataset), training, seed, epoch

    def __iter__(self) -> Iterator[int]:
        if not self.training:
            yield from range(self.size)
        else:
            generator = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self.seed, self.epoch])))
            yield from map(int, generator.permutation(self.size))

    def __len__(self) -> int:
        return self.size


def window_loader(dataset: ArticleWindows, batch_size: int, *, training: bool = False,
                  seed: int = 13, epoch: int = 0) -> DataLoader:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")
    # A private generator also prevents DataLoader's base-seed draw from perturbing model RNG.
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, drop_last=False, num_workers=0,
                      sampler=WindowOrder(dataset, training=training, seed=seed, epoch=epoch),
                      generator=generator)


def budget_mask(loss_mask: torch.Tensor, remaining_tokens: int) -> torch.Tensor:
    """Keep exactly the remaining valid targets in batch-major order, without mutation."""
    if loss_mask.dtype != torch.bool or loss_mask.ndim != 2 or remaining_tokens < 0:
        raise ValueError("Expected a boolean [batch, time] mask and a nonnegative token budget")
    ranks = loss_mask.reshape(-1).to(torch.long).cumsum(0).reshape_as(loss_mask)
    return loss_mask & (ranks <= remaining_tokens)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor,
                         loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return summed natural-log NLL and valid count; divide their sums across batches."""
    if (logits.ndim != 3 or logits.shape[:2] != targets.shape or targets.ndim != 2
            or loss_mask.shape != targets.shape or loss_mask.dtype != torch.bool
            or not logits.is_floating_point()
            or targets.dtype != torch.long or logits.device != targets.device
            or targets.device != loss_mask.device):
        raise ValueError("Expected colocated logits [B,T,V], long targets [B,T], bool mask [B,T]")
    count = loss_mask.sum()
    if not bool(count):
        raise ValueError("Cannot compute loss for zero prediction targets")
    # Select first: padding logits (even NaNs) cannot affect the objective or gradients.
    loss_sum = F.cross_entropy(logits[loss_mask].float(), targets[loss_mask], reduction="sum")
    return loss_sum, count


def initial_lstm_state(input_ids: torch.Tensor, layers: int, hidden_width: int,
                       *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Fresh hidden AND cell state [layers, batch, hidden]; call for every batch."""
    if input_ids.ndim != 2 or layers <= 0 or hidden_width <= 0 or not dtype.is_floating_point:
        raise ValueError("Invalid LSTM state shape/dtype")
    shape = (layers, input_ids.shape[0], hidden_width)
    return (torch.zeros(shape, dtype=dtype, device=input_ids.device),
            torch.zeros(shape, dtype=dtype, device=input_ids.device))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = (load_windows(args.output or args.tokens / "windows", args.tokens) if args.check
              else build_windows(args.tokens, args.output))
    print({split: row["windows"] for split, row in result["splits"].items()})


if __name__ == "__main__":
    main()
