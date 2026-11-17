"""Shared same-target context ablation (protocol evaluation.context.same_target).

The same late targets (positions 385-512 of every full window) are scored under
several histories, so only the context changes, never the target population:

    full        [BOS, x_1, ..., x_(t-1)]                      (prefix inside the window)
    k = 4/32/128 [BOS, x_(t-k), ..., x_(t-1)]
    shuffle     [BOS, permute(x_1..x_(t-129)), x_(t-128), ..., x_(t-1)]

This module is model-agnostic. It builds and persists the target manifest before
any scoring, rebuilds each condition's history deterministically, and summarizes
per-target NLL from any model. Histories returned here EXCLUDE BOS: each scorer
prepends its own start symbol (BOS for neural models; a null context for KenLM).

    python -m comparison.same_target --split validation   # build + persist manifest
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset.common import DEFAULT_DATA_ROOT, file_record, read_json, sha256_file, utc_now, write_json
from dataset.encode import split_arrays
from dataset.windows import WINDOW_TARGETS, load_windows
from .protocol import DEFAULT_PROTOCOL, load_protocol

FULL = "full"
SHUFFLE = "distant_shuffle_128"
DEFAULT_OUTPUT_ROOT = DEFAULT_DATA_ROOT.parent / "results" / "same_target"


@dataclass(frozen=True)
class Rules:
    first_position: int  # 1-based, inclusive
    last_position: int
    limits: tuple[int, ...]  # finite history limits, e.g. (4, 32, 128)
    local_tokens: int
    seed: int

    @property
    def conditions(self) -> tuple[str, ...]:
        return (FULL, *(f"k{limit}" for limit in self.limits), SHUFFLE)


def load_rules(protocol: Path = DEFAULT_PROTOCOL) -> Rules:
    """Read and check the parts of the protocol this implementation follows literally."""
    config = load_protocol(protocol)
    context = config["evaluation"]["context"]
    rules = context["same_target"]
    first, last = rules["targets"]["positions_one_based"]
    limits = context["history_limits"]
    shuffle = rules["distant_shuffle"]
    if (rules["targets"]["sampling"] != "exhaustive_no_subsampling" or rules["targets"]["eos_scored"] is not True
            or rules["histories"]["limits"] != "evaluation.context.history_limits" or limits[-1] != "full"
            or rules["histories"]["finite_limit_handling"] != "retain_exactly_k_nearest_preceding_tokens_in_original_order"
            or shuffle["rng"] != "numpy_PCG64_SeedSequence" or shuffle["draws_per_target"] != 1
            or shuffle["seed_components"] != ["seed", "article_index", "window_target_start_zero_based",
                                              "target_position_one_based"]):
        raise ValueError("same_target rules in the protocol differ from this implementation")
    finite = tuple(int(limit) for limit in limits[:-1])
    if not 1 <= first <= last <= WINDOW_TARGETS or first - 1 <= max(finite) or first - 1 <= shuffle["local_history_tokens"]:
        raise ValueError("Every target needs more preceding tokens than the largest limit")
    return Rules(first, last, finite, int(shuffle["local_history_tokens"]), int(shuffle["seed"]))


def distant_permutation(rules: Rules, article: int, window_start: int, position: int) -> np.ndarray:
    """Permutation of the distant indices 0..(t-1-local)-1 for 1-based target position t."""
    distant = position - 1 - rules.local_tokens
    generator = np.random.Generator(np.random.PCG64(
        np.random.SeedSequence([rules.seed, int(article), int(window_start), int(position)])))
    return generator.permutation(distant)


def history(window: np.ndarray, position: int, condition: str, rules: Rules,
            permutation: np.ndarray | None = None) -> np.ndarray:
    """Preceding token IDs for 1-based target position (no BOS, never the target or future)."""
    prefix = np.asarray(window[:position - 1])
    if condition == FULL:
        return prefix
    if condition == SHUFFLE:
        if permutation is None:
            raise ValueError("The shuffle condition needs its persisted permutation")
        distant = len(prefix) - rules.local_tokens
        return np.concatenate([prefix[:distant][permutation], prefix[distant:]])
    limit = int(condition.removeprefix("k"))
    if condition != f"k{limit}" or limit not in rules.limits:
        raise ValueError(f"Unknown condition: {condition}")
    return prefix[-limit:]


def build_manifest(split: str, output: Path, *, token_root: Path, window_root: Path,
                   protocol: Path = DEFAULT_PROTOCOL) -> dict:
    """Persist ordered target identities and IDs (and shuffle provenance) before scoring."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    rules = load_rules(protocol)
    load_windows(window_root, token_root)
    tokens, offsets = split_arrays(token_root, split)
    index = np.load(Path(window_root) / f"{split}.windows.npy", allow_pickle=False)
    full = index[index[:, 2] == WINDOW_TARGETS]
    positions = np.arange(rules.first_position, rules.last_position + 1)

    rows, unchanged, digest = [], 0, hashlib.sha256()
    for article, start, _ in full:
        window = np.asarray(tokens[int(offsets[article]) + int(start):][:WINDOW_TARGETS], dtype=np.int64)
        for position in positions:
            permutation = distant_permutation(rules, article, start, position)
            digest.update(permutation.astype("<i4").tobytes())
            distant = window[:position - 1 - rules.local_tokens]
            unchanged += bool(np.array_equal(distant[permutation], distant))
            rows.append((article, start, position, window[position - 1]))
    targets = np.asarray(rows, dtype="<i8").reshape(-1, 4)
    short = index[index[:, 2] != WINDOW_TARGETS]

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        np.save(staging / "targets.npy", targets, allow_pickle=False)
        manifest = {
            "schema_version": 1, "status": "PASS", "created_at": utc_now(), "split": split,
            "target_columns": ["article_index", "window_target_start_zero_based", "target_position_one_based",
                               "target_id"],
            "conditions": list(rules.conditions), "history_excludes_bos": True,
            "rules": {"positions_one_based": [rules.first_position, rules.last_position],
                      "finite_limits": list(rules.limits), "local_history_tokens": rules.local_tokens,
                      "shuffle_seed": rules.seed, "rng": "numpy_PCG64_SeedSequence"},
            "shuffle": {"permutations_sha256": digest.hexdigest(), "unchanged_token_sequences": int(unchanged)},
            "coverage": {"included_articles": int(len(np.unique(full[:, 0]))), "included_windows": int(len(full)),
                         "included_targets": int(len(targets)), "excluded_short_windows": int(len(short)),
                         "excluded_short_window_targets": int(short[:, 2].sum())},
            "provenance": {"protocol_sha256": sha256_file(protocol),
                           "tokenizer_sha256": read_json(Path(token_root) / "manifest.json")["tokenizer_sha256"],
                           "token_manifest_sha256": sha256_file(Path(token_root) / "manifest.json"),
                           "window_manifest_sha256": sha256_file(Path(window_root) / "manifest.json"),
                           "code_sha256": sha256_file(Path(__file__)), "numpy_version": np.__version__},
            "targets_file": file_record(staging / "targets.npy", staging),
        }
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    return manifest


def load_manifest(root: Path, protocol: Path = DEFAULT_PROTOCOL) -> tuple[dict, np.ndarray, Rules]:
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    record = manifest["targets_file"]
    if (manifest.get("status") != "PASS" or sha256_file(root / record["path"]) != record["sha256"]
            or manifest["provenance"]["protocol_sha256"] != sha256_file(protocol)):
        raise ValueError(f"Same-target manifest is invalid or bound to another protocol: {root}")
    return manifest, np.load(root / record["path"], allow_pickle=False), load_rules(protocol)


def iter_target_histories(targets: np.ndarray, tokens: np.ndarray, offsets: np.ndarray, rules: Rules):
    """Yield (row, target_id, {condition: history}) in manifest order."""
    for row, (article, start, position, target) in enumerate(targets):
        window = np.asarray(tokens[int(offsets[article]) + int(start):][:WINDOW_TARGETS], dtype=np.int64)
        if int(window[position - 1]) != int(target):
            raise ValueError("Token data differ from the persisted target manifest")
        permutation = distant_permutation(rules, article, start, position)
        yield row, int(target), {condition: history(window, int(position), condition, rules, permutation)
                                 for condition in rules.conditions}


def summarize_scores(nll: dict[str, np.ndarray], conditions: tuple[str, ...]) -> dict:
    """Token-weighted condition NLL and gains (positive = intact full history helps)."""
    full = nll[FULL]
    if any(len(nll[c]) != len(full) or not np.isfinite(nll[c]).all() for c in conditions):
        raise ValueError("Missing or non-finite same-target scores")
    summary = {"targets": int(len(full)), "condition_nll": {c: float(nll[c].mean()) for c in conditions}}
    summary["truncation_gain"] = {c: float((nll[c] - full).mean()) for c in conditions if c.startswith("k")}
    summary["shuffle_gain"] = float((nll[SHUFFLE] - full).mean())
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--windows", type=Path, help="Default: <tokens>/windows")
    parser.add_argument("--output", type=Path, help="Default: ../results/same_target/manifest_<split>")
    args = parser.parse_args()
    manifest = build_manifest(args.split, args.output or DEFAULT_OUTPUT_ROOT / f"manifest_{args.split}",
                              token_root=args.tokens, window_root=args.windows or args.tokens / "windows")
    print({"split": args.split, **manifest["coverage"], **manifest["shuffle"]})


if __name__ == "__main__":
    main()
