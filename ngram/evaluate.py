"""Evaluate the 5-gram on the shared 512-target windows and on BLiMP.

Window metrics follow comparison/protocol.yaml (evaluation.perplexity and .context):
perplexity over all non-padding targets; the position curve and context gain use full
windows only (short windows are counted and reported as exclusions). Each window is
scored from a fresh <s>, so the 5-gram can use at most four previous targets.

    python -m ngram.evaluate --split validation            # sanity check
    python -m ngram.evaluate --split test --blimp          # final reporting only
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import numpy as np

from comparison.blimp_scorer import evaluate_blimp
from comparison.protocol import DEFAULT_PROTOCOL, load_protocol
from dataset.common import DEFAULT_DATA_ROOT, file_record, read_json, sha256_file, utc_now, write_json
from dataset.encode import split_arrays
from dataset.windows import WINDOW_TARGETS, load_windows
from .lm import DEFAULT_MODEL_ROOT, KenLMScorer, TargetLogProbs, window_scorer
from .train import DEFAULT_LABEL


def score_split(target_log_probs: TargetLogProbs, tokens: np.ndarray, offsets: np.ndarray,
                index: np.ndarray) -> dict[str, np.ndarray]:
    """NLL per target [windows, 512] (NaN for padding) and OOV counts per window."""
    nll = np.full((len(index), WINDOW_TARGETS), np.nan, dtype=np.float64)
    oov = np.zeros(len(index), dtype=np.int64)
    for row, (article, local, valid) in enumerate(np.asarray(index, dtype=np.int64)):
        start = int(offsets[article]) + int(local)
        targets = np.asarray(tokens[start:start + valid], dtype=np.int64).tolist()
        log_probs, oov[row] = target_log_probs(targets)
        nll[row, :valid] = -log_probs
    return {"nll": nll, "oov": oov}


def summarize_windows(nll: np.ndarray, oov: np.ndarray, index: np.ndarray, context: dict) -> dict:
    valid = index[:, 2]
    total_targets = int(valid.sum())
    total_nll = float(np.nansum(nll))
    if not total_targets or not np.isfinite(total_nll):
        raise ValueError("Cannot summarize: no targets or non-finite NLL")
    full = valid == WINDOW_TARGETS
    curve = nll[full].mean(axis=0)  # Mean NLL at each 1-based position over full windows.
    early_lo, early_hi = context["early_positions_one_based"]
    late_lo, late_hi = context["late_positions_one_based"]
    early, late = float(curve[early_lo - 1:early_hi].mean()), float(curve[late_lo - 1:late_hi].mean())
    return {
        "targets": total_targets, "windows": len(index), "mean_nll": total_nll / total_targets,
        "perplexity": math.exp(total_nll / total_targets),
        "oov_targets": int(oov.sum()), "oov_rate": float(oov.sum() / total_targets),
        "context": {
            "full_windows": int(full.sum()), "excluded_short_windows": int((~full).sum()),
            "early_positions_one_based": [early_lo, early_hi], "late_positions_one_based": [late_lo, late_hi],
            "early_mean_nll": early, "late_mean_nll": late, "context_gain": early - late,
            "position_mean_nll_first_8": curve[:8].tolist(),
        },
        "position_mean_nll": curve.tolist(),
    }


def evaluate_windows(target_log_probs: TargetLogProbs, output: Path, *, split: str, token_root: Path,
                     window_root: Path, protocol: Path, model_label: str, model_manifest: Path | None) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite results: {output}")
    load_windows(window_root, token_root)  # Verifies hashes and exact coverage before scoring.
    tokens, offsets = split_arrays(token_root, split)
    index = np.load(Path(window_root) / f"{split}.windows.npy", allow_pickle=False)
    scored = score_split(target_log_probs, tokens, offsets, index)
    summary = summarize_windows(scored["nll"], scored["oov"], index, load_protocol(protocol)["evaluation"]["context"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        # Per-window detail supports article-level bootstrap and model-disagreement analyses.
        np.savez_compressed(staging / "windows.npz", index=index, nll=scored["nll"], oov=scored["oov"])
        result = {
            "schema_version": 1, "status": "PASS", "created_at": utc_now(), "model_label": model_label,
            "split": split, "protocol_sha256": sha256_file(protocol),
            "token_manifest_sha256": sha256_file(Path(token_root) / "manifest.json"),
            "window_manifest_sha256": sha256_file(Path(window_root) / "manifest.json"),
            "model_manifest_sha256": sha256_file(model_manifest) if model_manifest else None,
            "history": "fresh_<s>_each_window_at_most_4_previous_targets",
            "windows_file": file_record(staging / "windows.npz", staging), **summary,
        }
        write_json(staging / "summary.json", result)
        staging.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_ROOT / DEFAULT_LABEL)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--blimp", action="store_true", help="Also score BLiMP (final evaluation only)")
    parser.add_argument("--tokens", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--windows", type=Path, help="Default: <tokens>/windows")
    parser.add_argument("--results", type=Path, default=DEFAULT_MODEL_ROOT.parent.parent / "results" / "ngram")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()

    model_manifest = args.model_dir / "manifest.json"
    if read_json(model_manifest).get("status") != "PASS":
        raise ValueError(f"Model manifest is not PASS: {model_manifest}")
    label = args.model_dir.name
    scorer = KenLMScorer(args.model_dir / "model.bin")
    summary = evaluate_windows(
        scorer.target_log_probs, args.results / f"{label}_{args.split}", split=args.split,
        token_root=args.tokens, window_root=args.windows or args.tokens / "windows",
        protocol=args.protocol, model_label=label, model_manifest=model_manifest)
    print({key: summary[key] for key in ("split", "perplexity", "mean_nll", "oov_targets")},
          {"context_gain": summary["context"]["context_gain"]})
    if args.blimp:
        blimp = evaluate_blimp(window_scorer(scorer.target_log_probs), args.results / f"{label}_blimp",
                               model_label=label, protocol=args.protocol)
        print({"blimp_overall_accuracy": blimp["primary"]["overall"]["accuracy"]})


if __name__ == "__main__":
    main()
