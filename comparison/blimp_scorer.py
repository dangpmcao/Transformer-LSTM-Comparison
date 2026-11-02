"""Score BLiMP minimal pairs by comparing full-sentence log probabilities.

Accuracy follows Warstadt et al. (2020, TACL 8:377-392, section 5): the proportion
of pairs whose acceptable sentence receives the higher probability. BOS/EOS, ties,
long sentences, and sensitivity analyses follow evaluation.blimp in protocol.yaml.

Any model plugs in through ScoreWindows; neural_window_scorer adapts neural models.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dataset.common import (
    DEFAULT_DATA_ROOT, SOURCE_LOCK, contained_path, file_record, read_json, sha256_file, utc_now, write_json,
)
from dataset.download import verify_file
from dataset.prepare import validate_blimp
from dataset.windows import WINDOW_TARGETS
from tokenizer.core import DEFAULT_OUTPUT, SPECIAL_IDS, encode_text, load_tokenizer
from .protocol import DEFAULT_PROTOCOL, load_protocol

# Target IDs for each window -> summed natural-log probability for each window.
# Every window starts from BOS with fresh model state; BOS itself is never scored.
ScoreWindows = Callable[[list[list[int]]], Sequence[float]]
# Batch fields as in NextTokenWindowDataset, padded to the longest window in the batch.
LogitsFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]

IMPLEMENTED_RULES = {
    "primary_score": "summed_subword_log_probability", "start_with_bos": True, "include_eos": True,
    "ties": "incorrect_including_identical_candidates",
    "overlength": "score_all_targets_in_independent_512_target_windows_and_report_count",
    "report": ["overall", "twelve_groups", "sixty_seven_paradigms"],
    "sensitivities": ["equal_bpe_length_pairs", "length_normalized_log_probability"],
    "truncate_or_drop": False,
}


def split_target_windows(targets: list[int]) -> list[list[int]]:
    """Split one sentence's targets without dropping any, as dataset windows do for articles."""
    return [targets[start:start + WINDOW_TARGETS] for start in range(0, len(targets), WINDOW_TARGETS)]


def make_window_batch(windows: list[list[int]], device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Build shifted inputs and masks [batch, longest window]; causal models ignore the padding."""
    length = max(map(len, windows), default=0)
    if not windows or not 0 < min(map(len, windows)) <= length <= WINDOW_TARGETS:
        raise ValueError(f"Every window needs 1 to {WINDOW_TARGETS} targets")
    target_ids = torch.full((len(windows), length), SPECIAL_IDS["<pad>"], dtype=torch.long)
    input_ids = torch.full_like(target_ids, SPECIAL_IDS["<pad>"])
    loss_mask = torch.zeros_like(target_ids, dtype=torch.bool)
    for row, window in enumerate(windows):
        valid = len(window)
        target_ids[row, :valid] = torch.tensor(window, dtype=torch.long)
        input_ids[row, 0] = SPECIAL_IDS["<bos>"]
        input_ids[row, 1:valid] = target_ids[row, :valid - 1]
        loss_mask[row, :valid] = True
    batch = {
        "input_ids": input_ids, "target_ids": target_ids, "loss_mask": loss_mask,
        "attention_mask": loss_mask.clone(), "position_ids": torch.arange(length).repeat(len(windows), 1),
    }
    return {name: tensor.to(device) for name, tensor in batch.items()}


def sum_log_probabilities(logits: torch.Tensor, targets: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Return one summed natural-log probability per row, on CPU, in float64."""
    if (logits.ndim != 3 or logits.shape[:2] != targets.shape or loss_mask.shape != targets.shape
            or loss_mask.dtype != torch.bool or not logits.is_floating_point()):
        raise ValueError("Expected logits [B,T,V], targets [B,T], and a bool mask [B,T]")
    # Select first: padding logits (even NaNs) cannot affect the scores, as in masked_cross_entropy.
    values = -F.cross_entropy(logits[loss_mask].float(), targets[loss_mask], reduction="none").cpu().double()
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite log probability for a scored target")
    return torch.zeros(len(targets), dtype=torch.float64).index_add_(0, loss_mask.nonzero()[:, 0].cpu(), values)


def neural_window_scorer(logits_fn: LogitsFn, *, device: torch.device | str = "cpu",
                         batch_size: int = 64) -> ScoreWindows:
    """Adapt a model; logits_fn must run in eval mode and create fresh LSTM states for each call."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive")

    def score(windows: list[list[int]]) -> list[float]:
        if isinstance(logits_fn, torch.nn.Module) and logits_fn.training:
            raise ValueError("Call model.eval() first: dropout must not affect BLiMP scores")
        scores = [0.0] * len(windows)
        # Similar lengths share a batch, so less time goes to padding.
        order = sorted(range(len(windows)), key=lambda index: len(windows[index]))
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                chosen = order[start:start + batch_size]
                batch = make_window_batch([windows[index] for index in chosen], device)
                totals = sum_log_probabilities(logits_fn(batch), batch["target_ids"], batch["loss_mask"])
                for index, total in zip(chosen, totals.tolist(), strict=True):
                    scores[index] = total
        return scores

    return score


def load_pairs(data_root: Path = DEFAULT_DATA_ROOT, source_lock: Path = SOURCE_LOCK) -> list[dict]:
    """Verify the pinned downloads, then return every pair with its 12-group label, in source order."""
    root = Path(data_root).resolve()
    lock = read_json(source_lock)
    for specification in lock["files"]:
        if specification["source"] == "blimp":
            verify_file(contained_path(root, specification["path"]), specification)
    validate_blimp(root, lock["blimp"], save_mapping=False)
    pairs = []
    for entry in sorted(read_json(root / "processed/blimp/paradigm_groups.json"), key=lambda row: row["paradigm"]):
        path = contained_path(root, f"raw/blimp/data/{entry['paradigm']}.jsonl")
        with path.open(encoding="utf-8") as stream:
            items = [json.loads(line) for line in stream]
        if len(items) != entry["pairs"] or any(
                item["linguistics_term"] != entry["source_linguistics_term"] for item in items):
            raise ValueError(f"Stale BLiMP group mapping; rerun python -m dataset.prepare: {path.name}")
        pairs.extend({
            "paradigm": entry["paradigm"], "pair_id": str(item["pairID"]), "group": entry["group"],
            "sentence_good": item["sentence_good"], "sentence_bad": item["sentence_bad"],
        } for item in items)
    return pairs


def score_pairs(pairs: list[dict], tokenizer: Tokenizer, score_windows: ScoreWindows) -> list[dict]:
    """Return one result per pair; ties count as incorrect because comparisons are strict."""
    encoded = [(encode_text(tokenizer, pair["sentence_good"], add_eos=True),
                encode_text(tokenizer, pair["sentence_bad"], add_eos=True)) for pair in pairs]
    # Score each distinct window once: identical candidates then tie exactly, whatever the batching.
    unique = list(dict.fromkeys(tuple(window) for sentences in encoded for targets in sentences
                                for window in split_target_windows(targets)))
    scores = score_windows([list(window) for window in unique])
    if len(scores) != len(unique):
        raise ValueError(f"Expected {len(unique)} window scores, got {len(scores)}")
    by_window = dict(zip(unique, map(float, scores), strict=True))
    if not all(map(math.isfinite, by_window.values())):
        raise ValueError("Non-finite BLiMP window score")

    def total(targets: list[int]) -> float:
        return sum(by_window[tuple(window)] for window in split_target_windows(targets))

    results = []
    for pair, (good, bad) in zip(pairs, encoded, strict=True):
        good_score, bad_score = total(good), total(bad)
        results.append({
            "paradigm": pair["paradigm"], "pair_id": pair["pair_id"], "group": pair["group"],
            "good_log_probability": good_score, "bad_log_probability": bad_score,
            "good_targets": len(good), "bad_targets": len(bad), "identical_candidates": good == bad,
            "correct": good_score > bad_score,
            "length_normalized_correct": good_score / len(good) > bad_score / len(bad),
        })
    return results


def _accuracy(results: list[dict], decision: str) -> dict:
    correct = sum(row[decision] for row in results)
    return {"pairs": len(results), "correct": correct, "accuracy": correct / len(results) if results else None}


def _breakdown(results: list[dict], decision: str, groups: list[str], paradigms: list[str]) -> dict:
    return {
        "overall": _accuracy(results, decision),
        "groups": {name: _accuracy([row for row in results if row["group"] == name], decision) for name in groups},
        "paradigms": {name: _accuracy([row for row in results if row["paradigm"] == name], decision) for name in paradigms},
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-pair results; a level with no equal-length pairs reports accuracy None."""
    groups = sorted({row["group"] for row in results})
    paradigms = sorted({row["paradigm"] for row in results})
    equal_length = [row for row in results if row["good_targets"] == row["bad_targets"]]
    return {
        "primary": _breakdown(results, "correct", groups, paradigms),
        "sensitivities": {
            "equal_bpe_length_pairs": _breakdown(equal_length, "correct", groups, paradigms),
            "length_normalized_log_probability": _breakdown(results, "length_normalized_correct", groups, paradigms),
        },
        "counts": {
            "tied_pairs": sum(row["good_log_probability"] == row["bad_log_probability"] for row in results),
            "identical_candidate_pairs": sum(row["identical_candidates"] for row in results),
            "overlength_sentences": sum((row["good_targets"] > WINDOW_TARGETS) + (row["bad_targets"] > WINDOW_TARGETS)
                                        for row in results),
        },
    }


def evaluate_blimp(score_windows: ScoreWindows, output: Path, *, model_label: str,
                   data_root: Path = DEFAULT_DATA_ROOT, tokenizer_dir: Path = DEFAULT_OUTPUT,
                   protocol: Path = DEFAULT_PROTOCOL, source_lock: Path = SOURCE_LOCK) -> dict:
    """Publish pairs.jsonl and summary.json in a new folder; BLiMP must not select models."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite BLiMP results: {output}")
    if not model_label:
        raise ValueError("A model label is required to identify these results")
    config = load_protocol(protocol)
    if config["evaluation"]["blimp"] != IMPLEMENTED_RULES:
        raise ValueError("BLiMP rules in the protocol differ from the implemented scorer")
    tokenizer_sha256 = sha256_file(Path(tokenizer_dir) / "tokenizer.json")
    if config["inputs"]["tokenizer_sha256"] != tokenizer_sha256:
        raise ValueError("Tokenizer differs from the protocol's shared tokenizer")
    results = score_pairs(load_pairs(data_root, source_lock), load_tokenizer(tokenizer_dir), score_windows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        with (staging / "pairs.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            stream.writelines(json.dumps(row, sort_keys=True) + "\n" for row in results)
        summary = {
            "schema_version": 1, "status": "PASS", "created_at": utc_now(), "model_label": model_label,
            "protocol_sha256": sha256_file(protocol), "tokenizer_sha256": tokenizer_sha256,
            "source": read_json(source_lock)["blimp"], "code_sha256": sha256_file(Path(__file__)),
            "pairs_file": file_record(staging / "pairs.jsonl", staging), **summarize(results),
        }
        write_json(staging / "summary.json", summary)
        if output.exists():
            raise FileExistsError(f"Output appeared during BLiMP scoring: {output}")
        staging.rename(output)
    return summary
