"""Check a real training article through encoding, batching, and masked gradients."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from tokenizer.core import DEFAULT_OUTPUT, encode_text, load_tokenizer, verified_article_path, verified_data_manifest
from .common import DEFAULT_DATA_ROOT, sha256_file, utc_now, write_json
from .encode import load_cache
from .windows import ArticleWindows, initial_lstm_state, masked_cross_entropy


def check_inputs(data_root: Path, token_root: Path, artifacts: Path = DEFAULT_OUTPUT,
                 window_root: Path | None = None) -> dict:
    tokenizer = load_tokenizer(artifacts)
    cache = load_cache(token_root)
    if cache["tokenizer_sha256"] != sha256_file(Path(artifacts) / "tokenizer.json"):
        raise ValueError("Sample tokenizer differs from token cache")
    data = verified_data_manifest(data_root)
    path = verified_article_path(data_root, data, "train")
    with path.open(encoding="utf-8") as stream:
        article = json.loads(next(stream))
    expected = encode_text(tokenizer, article["text"], add_eos=True)
    dataset = ArticleWindows(token_root, "train", window_root)
    end = int(dataset.offsets[1])
    if not np.array_equal(dataset.tokens[:end], expected):
        raise ValueError("Sample article tokenization differs from persisted IDs")
    indices = np.flatnonzero(dataset.index[:, 0] == 0)
    reconstructed = []
    for index in indices:
        item = dataset[int(index)]
        reconstructed.extend(item["target_ids"][item["loss_mask"]].tolist())
    if reconstructed != expected:
        raise ValueError("Window targets fail to reconstruct the sample article plus EOS")
    for batch_size in (1, 2):
        items = [dataset[int(indices[-1])]]
        if batch_size == 2:
            items.append(dataset[0])
        batch = default_collate(items)
        shape = (batch_size, 512)
        if (batch["input_ids"].shape != shape or batch["input_ids"].dtype != torch.long
                or not torch.equal(batch["position_ids"], torch.arange(512).expand(batch_size, -1))):
            raise ValueError("Unexpected batch shape, dtype, or positions")
        logits = torch.zeros((*shape, cache["vocab_size"]), dtype=torch.float32)
        logits[~batch["loss_mask"]] = float("nan")
        logits.requires_grad_()
        loss_sum, count = masked_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        mean = loss_sum / count
        if not torch.isclose(mean, torch.tensor(math.log(cache["vocab_size"])), atol=1e-5):
            raise ValueError("Masked loss is not the expected uniform-vocabulary NLL")
        mean.backward()
        if not torch.isfinite(logits.grad).all() or torch.count_nonzero(logits.grad[~batch["loss_mask"]]):
            raise ValueError("Padding affected loss gradients")
        hidden, cell = initial_lstm_state(batch["input_ids"], 2, 4, dtype=torch.float32)
        if hidden.shape != (2, batch_size, 4) or hidden.any() or cell.any():
            raise ValueError("LSTM window reset failed")
    return {
        "status": "PASS", "checked_at": utc_now(), "article_id": article["article_id"],
        "article_prediction_tokens": len(expected), "article_windows": len(indices),
        "batch_sizes": [1, 2], "device": "cpu", "logit_dtype": "float32",
        "masked_loss_and_gradients": "PASS", "lstm_hidden_and_cell_reset": "PASS",
        "token_manifest_sha256": sha256_file(Path(token_root) / "manifest.json"),
        "window_manifest_sha256": sha256_file(dataset.window_root / "manifest.json"),
        "scope": "Data/objective smoke test only; no neural model training or test-set scoring.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    tokens = args.tokens or args.data_root / "processed/tokens"
    result = check_inputs(args.data_root, tokens, args.artifacts, args.windows)
    write_json(args.output or tokens / "checks.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
