"""Score the shared Copy(L) manifest with the 5-gram (protocol evaluation.copy).

Each input starts from KenLM's null context in place of BOS. A 5-gram sees at most
four previous tokens, so it can only "copy" at the first positions of the repeated
span (where A's last tokens precede A's first) and from repeats already inside A:
its copy gain is the no-long-context reference for the neural models.

Random-ID spans can contain the 99 IDs never seen in train; they are scored as
KenLM <unk> (declared before scoring, never dropped) and counted per source and L.

    python -m comparison.copy_task      # once, shared by all models
    python -m ngram.copy_task
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from comparison.copy_task import DEFAULT_OUTPUT, load_manifest, score_spans, summarize
from comparison.protocol import DEFAULT_PROTOCOL
from dataset.common import file_record, read_json, sha256_file, utc_now, write_json
from .lm import DEFAULT_MODEL_ROOT, KenLMScorer
from .train import DEFAULT_LABEL


def evaluate(scorer: KenLMScorer, manifest_root: Path, output: Path, *, model_label: str,
             model_manifest: Path, protocol: Path = DEFAULT_PROTOCOL) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite results: {output}")
    manifest, spans = load_manifest(manifest_root, protocol)

    def sequence_nll(targets):
        return -scorer.target_log_probs(targets)[0]

    results, arrays = {}, {}
    for (source, length), span in sorted(spans.items()):
        nll = score_spans(sequence_nll, span)
        oov = sum(not scorer.knows(int(token)) for token in np.concatenate([span["a"], span["b"]]).ravel())
        results.setdefault(source, {})[str(length)] = {**summarize(nll), "oov_span_tokens": int(oov)}
        arrays.update({f"{source}_L{length}_{condition}": values for condition, values in nll.items()})
    primary = results["test_natural_text"][str(manifest["primary_length"])]["all_positions"]["copy_gain"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        np.savez_compressed(staging / "per_target_nll.npz", **arrays)
        summary = {
            "schema_version": 1, "status": "PASS", "created_at": utc_now(), "model_label": model_label,
            "split": manifest["split"], "start_symbol": "kenlm_null_context_in_place_of_bos",
            "oov_treatment": "kenlm_unk_never_dropped",
            f"natural_copy_{manifest['primary_length']}": primary, "results": results,
            "provenance": {**manifest["provenance"], "manifest_sha256": sha256_file(Path(manifest_root) / "manifest.json"),
                           "model_manifest_sha256": sha256_file(model_manifest),
                           "scorer_code_sha256": sha256_file(Path(__file__))},
            "scores_file": file_record(staging / "per_target_nll.npz", staging),
        }
        write_json(staging / "summary.json", summary)
        staging.rename(output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_ROOT / DEFAULT_LABEL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_OUTPUT.parent)
    args = parser.parse_args()
    model_manifest = args.model_dir / "manifest.json"
    if read_json(model_manifest).get("status") != "PASS":
        raise ValueError(f"Model manifest is not PASS: {model_manifest}")
    label = args.model_dir.name
    summary = evaluate(KenLMScorer(args.model_dir / "model.bin"), args.manifest, args.results / f"{label}_test",
                       model_label=label, model_manifest=model_manifest)
    print({key: value for key, value in summary.items() if key.startswith("natural_copy")})
    for source, by_length in summary["results"].items():
        for length, row in by_length.items():
            print(source, length, row["all_positions"], row["exclude_first_four"], "oov", row["oov_span_tokens"])


if __name__ == "__main__":
    main()
