"""Shared Copy(L) span manifest and metrics (protocol evaluation.copy).

Each example holds two independent spans A and B of L token IDs from one source:

    repeat   [BOS, A_1..A_L, A_1..A_L]   copy gain    = mean NLL(first A) - mean NLL(second A)
    control  [BOS, A_1..A_L, B_1..B_L]   control gain = mean NLL(A) - mean NLL(B)

The control has the same length and positions without a repetition, so
"copy gain minus control gain" isolates copying from position effects. Both gains
are also reported without the first four positions of each span (the 5-gram can
copy those from local history alone). EOS is never scored: natural spans never
contain or cross an EOS, and nothing is appended. With L <= 200 every input fits in
one 512-target window, scored from fresh model state.

This module is model-agnostic: it persists every span before any scoring and
summarizes per-target NLL from any model. Sequences returned here EXCLUDE BOS; each
scorer prepends its own start symbol (BOS for neural models, null context for KenLM).

    python -m comparison.copy_task      # build + persist the manifest (test split)
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from dataset.common import DEFAULT_DATA_ROOT, file_record, read_json, sha256_file, utc_now, write_json
from dataset.encode import split_arrays
from dataset.windows import WINDOW_TARGETS
from tokenizer.core import SPECIAL_IDS
from .protocol import DEFAULT_PROTOCOL, load_protocol

NATURAL, RANDOM = "test_natural_text", "uniform_nonspecial_token_ids"
SOURCES = (NATURAL, RANDOM)
CONDITIONS = ("repeat", "control")
SKIP_POSITIONS = 4  # controls: exclude_first_four_positions_of_each_span
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT.parent / "results" / "copy" / "manifest_test"
# Sequence of target IDs (no BOS) -> natural-log NLL per target, from fresh state.
SequenceNLL = Callable[[Sequence[int]], np.ndarray]


def load_rules(protocol: Path = DEFAULT_PROTOCOL) -> dict:
    """Read and check the parts of the protocol this implementation follows literally."""
    rules = load_protocol(protocol)["evaluation"]["copy"]
    if (rules["sources"] != list(SOURCES) or rules["eos_scored"] is not False
            or rules["metric"] != "first_span_mean_nll_minus_repeated_span_mean_nll"
            or rules["natural_sampling"] != "uniform_valid_content_start_with_replacement_never_cross_article_or_eos"
            or rules["random_ids"] != "uniform_integers_4_through_16383"
            or rules["controls"] != ["independent_nonrepeat_span", "exclude_first_four_positions_of_each_span"]
            or rules["primary_length"] not in rules["lengths"] or 2 * max(rules["lengths"]) > WINDOW_TARGETS):
        raise ValueError("copy rules in the protocol differ from this implementation")
    return rules


def generator(seed: int, source: str, length: int) -> np.random.Generator:
    """One independent stream per (source, L): adding a length never changes the others."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, SOURCES.index(source), length])))


def valid_starts(offsets: np.ndarray, length: int) -> np.ndarray:
    """Every start whose span of `length` content tokens stays inside one article, before its EOS."""
    offsets = np.asarray(offsets, dtype=np.int64)
    # Article a holds tokens offsets[a]..offsets[a+1]-1, the last being its EOS.
    pieces = [np.arange(start, end - length) for start, end in zip(offsets[:-1], offsets[1:]) if end - 1 - start >= length]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype=np.int64)


def sample_spans(source: str, length: int, count: int, seed: int, tokens: np.ndarray,
                 offsets: np.ndarray) -> dict[str, np.ndarray]:
    """Spans A and B [count, L] and their start offsets in the split (-1 for random IDs)."""
    rng = generator(seed, source, length)
    if source == RANDOM:
        ids = rng.integers(4, 16384, size=(2, count, length), dtype=np.int64)
        starts = np.full((2, count), -1, dtype=np.int64)
    else:
        candidates = valid_starts(offsets, length)
        starts = candidates[rng.integers(0, len(candidates), size=(2, count))]
        ids = np.asarray(tokens, dtype=np.int64)[starts[..., None] + np.arange(length)]
    if (ids == SPECIAL_IDS["<eos>"]).any() or (ids < 4).any():
        raise ValueError("A copy span contains a control ID")
    return {"a": ids[0], "b": ids[1], "a_start": starts[0], "b_start": starts[1]}


def build_manifest(output: Path, *, token_root: Path, split: str = "test",
                   protocol: Path = DEFAULT_PROTOCOL) -> dict:
    """Persist every span, its token IDs and the sampling seed before any scoring."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {output}")
    rules = load_rules(protocol)
    if split != "test":
        raise ValueError("The protocol's natural source is test_natural_text")
    tokens, offsets = split_arrays(token_root, split)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        files = {}
        for source in SOURCES:
            for length in rules["lengths"]:
                name = f"{source}_L{length}.npz"
                np.savez_compressed(staging / name, **sample_spans(
                    source, length, rules["examples_per_length_per_source"], rules["seed"], tokens, offsets))
                files[f"{source}/{length}"] = file_record(staging / name, staging)
        manifest = {
            "schema_version": 1, "status": "PASS", "created_at": utc_now(), "split": split,
            "sources": list(SOURCES), "lengths": rules["lengths"], "primary_length": rules["primary_length"],
            "examples_per_length_per_source": rules["examples_per_length_per_source"],
            "sampling": {"seed": rules["seed"], "rng": "numpy_PCG64_SeedSequence",
                         "seed_components": ["seed", "source_index", "length"],
                         "draw_order": "a_then_b_in_one_call", "natural": rules["natural_sampling"],
                         "random_ids": rules["random_ids"]},
            "inputs": {"repeat": "[BOS] + A + A", "control": "[BOS] + A + B", "eos_scored": False},
            "provenance": {"protocol_sha256": sha256_file(protocol),
                           "tokenizer_sha256": read_json(Path(token_root) / "manifest.json")["tokenizer_sha256"],
                           "token_manifest_sha256": sha256_file(Path(token_root) / "manifest.json"),
                           "code_sha256": sha256_file(Path(__file__)), "numpy_version": np.__version__},
            "files": files,
        }
        write_json(staging / "manifest.json", manifest)
        staging.rename(output)
    return manifest


def load_manifest(root: Path, protocol: Path = DEFAULT_PROTOCOL) -> tuple[dict, dict[tuple[str, int], dict]]:
    """Verify hashes and the protocol's copy rules, then return all spans by (source, L)."""
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    rules = load_rules(protocol)
    if (manifest.get("status") != "PASS" or manifest["lengths"] != rules["lengths"]
            or manifest["sampling"]["seed"] != rules["seed"]):
        raise ValueError(f"Copy manifest is invalid or differs from the protocol: {root}")
    spans = {}
    for key, record in manifest["files"].items():
        if sha256_file(root / record["path"]) != record["sha256"]:
            raise ValueError(f"Copy manifest file changed: {record['path']}")
        source, length = key.split("/")
        with np.load(root / record["path"], allow_pickle=False) as data:
            spans[source, int(length)] = {name: data[name] for name in data.files}
    return manifest, spans


def score_spans(sequence_nll: SequenceNLL, spans: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Per-target NLL [examples, 2L] for the repeat and control inputs."""
    a, b = spans["a"], spans["b"]
    nll = {condition: np.empty((len(a), 2 * a.shape[1]), dtype=np.float64) for condition in CONDITIONS}
    for row in range(len(a)):
        nll["repeat"][row] = sequence_nll(np.concatenate([a[row], a[row]]).tolist())
        nll["control"][row] = sequence_nll(np.concatenate([a[row], b[row]]).tolist())
    if not all(np.isfinite(values).all() for values in nll.values()):
        raise ValueError("Non-finite copy NLL")
    return nll


def span_gain(nll: np.ndarray, skip: int = 0) -> float:
    """Mean NLL of the first span minus mean NLL of the second (positive = second is easier)."""
    length = nll.shape[1] // 2
    return float(nll[:, skip:length].mean() - nll[:, length + skip:].mean())


def summarize(nll: dict[str, np.ndarray]) -> dict:
    result = {}
    for label, skip in (("all_positions", 0), ("exclude_first_four", SKIP_POSITIONS)):
        copy, control = span_gain(nll["repeat"], skip), span_gain(nll["control"], skip)
        result[label] = {"copy_gain": copy, "control_gain": control, "copy_minus_control": copy - control}
    length = nll["repeat"].shape[1] // 2
    result["mean_nll"] = {"first_span": float(nll["repeat"][:, :length].mean()),
                          "repeated_span": float(nll["repeat"][:, length:].mean()),
                          "control_second_span": float(nll["control"][:, length:].mean())}
    result["examples"] = int(len(nll["repeat"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest(args.output, token_root=args.tokens)
    print({"output": str(args.output), "files": len(manifest["files"]), "seed": manifest["sampling"]["seed"]})


if __name__ == "__main__":
    main()
