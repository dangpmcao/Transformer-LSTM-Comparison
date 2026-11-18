"""5-gram adapter for the same-target context ablation (protocol five_gram_extension).

Scores every manifest target under each history condition and checks, PER TARGET,
that the 5-gram is invariant: with at least four preceding tokens in every
condition, it sees the same last four tokens, so all gains must be zero within
absolute_tolerance_nats. A failure here means the pipeline (not the model) leaks
or misaligns history, which is why the 5-gram runs on the same manifest.

    python -m ngram.same_target --split validation
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from comparison.protocol import DEFAULT_PROTOCOL, load_protocol
from comparison.same_target import (
    DEFAULT_OUTPUT_ROOT, FULL, iter_target_histories, load_manifest, summarize_scores,
)
from dataset.common import DEFAULT_DATA_ROOT, file_record, read_json, sha256_file, utc_now, write_json
from dataset.encode import split_arrays
from .lm import DEFAULT_MODEL_ROOT, KenLMScorer
from .train import DEFAULT_LABEL


def score_manifest(next_log_prob, targets: np.ndarray, tokens: np.ndarray, offsets: np.ndarray,
                   rules) -> dict[str, np.ndarray]:
    nll = {condition: np.empty(len(targets), dtype=np.float64) for condition in rules.conditions}
    for row, target, histories in iter_target_histories(targets, tokens, offsets, rules):
        for condition, prefix in histories.items():
            nll[condition][row] = -next_log_prob(prefix, target)
    return nll


def invariance(nll: dict[str, np.ndarray], tolerance: float) -> dict:
    deviation = {c: float(np.abs(values - nll[FULL]).max()) for c, values in nll.items() if c != FULL}
    failures = {c: int((np.abs(values - nll[FULL]) > tolerance).sum()) for c, values in nll.items() if c != FULL}
    return {"check": "per_target", "absolute_tolerance_nats": tolerance, "max_abs_deviation": deviation,
            "targets_outside_tolerance": failures, "status": "PASS" if not any(failures.values()) else "FAIL"}


def evaluate(scorer: KenLMScorer, manifest_root: Path, output: Path, *, token_root: Path, model_label: str,
             model_manifest: Path, protocol: Path = DEFAULT_PROTOCOL) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite results: {output}")
    manifest, targets, rules = load_manifest(manifest_root, protocol)
    tolerance = load_protocol(protocol)["evaluation"]["context"]["same_target"]["five_gram_extension"][
        "absolute_tolerance_nats"]
    tokens, offsets = split_arrays(token_root, manifest["split"])
    nll = score_manifest(scorer.next_log_prob, targets, tokens, offsets, rules)
    result = {**summarize_scores(nll, rules.conditions), "invariance": invariance(nll, tolerance)}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        np.savez_compressed(staging / "per_target_nll.npz", targets=targets, **nll)
        write_json(staging / "summary.json", {
            "schema_version": 1, "status": result["invariance"]["status"], "created_at": utc_now(),
            "model_label": model_label, "split": manifest["split"], "coverage": manifest["coverage"],
            "start_symbol": "kenlm_null_context_in_place_of_bos",
            "provenance": {**manifest["provenance"], "manifest_sha256": sha256_file(Path(manifest_root) / "manifest.json"),
                           "model_manifest_sha256": sha256_file(model_manifest),
                           "scorer_code_sha256": sha256_file(Path(__file__))},
            "scores_file": file_record(staging / "per_target_nll.npz", staging), **result,
        })
        staging.rename(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["validation", "test"], required=True)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_ROOT / DEFAULT_LABEL)
    parser.add_argument("--manifest", type=Path, help="Default: ../results/same_target/manifest_<split>")
    parser.add_argument("--tokens", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--results", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    model_manifest = args.model_dir / "manifest.json"
    if read_json(model_manifest).get("status") != "PASS":
        raise ValueError(f"Model manifest is not PASS: {model_manifest}")
    label = args.model_dir.name
    result = evaluate(KenLMScorer(args.model_dir / "model.bin"),
                      args.manifest or DEFAULT_OUTPUT_ROOT / f"manifest_{args.split}",
                      args.results / f"{label}_{args.split}", token_root=args.tokens,
                      model_label=label, model_manifest=model_manifest)
    print({"invariance": result["invariance"]["status"], "condition_nll": result["condition_nll"],
           "max_abs_deviation": result["invariance"]["max_abs_deviation"]})
    if result["invariance"]["status"] != "PASS":
        raise SystemExit("5-gram invariance check FAILED: history handling is wrong somewhere")


if __name__ == "__main__":
    main()
