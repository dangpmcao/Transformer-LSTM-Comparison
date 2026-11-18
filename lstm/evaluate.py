"""Final evaluation of trained LSTM checkpoints (test split, BLiMP, same-target, Copy(L)).

For each run and checkpoint view (protocol selection.report_views: best_validation =
best.pt, final_budget = final.pt) this writes, under <results>/<label>/<view>/:

    validation_check.json   recomputed validation NLL vs the training record (gate)
    test/                   perplexity, position curve, context gain (as ngram.evaluate)
    blimp/                  comparison.blimp_scorer with neural_window_scorer
    same_target/            protocol evaluation.context.same_target, shared manifest
    copy/                   protocol evaluation.copy, shared manifest

Scoring is strict fp32 (no autocast, TF32 off), in eval mode, and every input starts
from fresh zero LSTM states (LSTMLM.forward never carries state between calls).
The validation gate runs first: if the recomputed NLL differs from the training
record by more than 1e-4 nats, nothing on the test split is scored for that view.
Existing outputs are never overwritten; a rerun skips finished parts.

    python -m lstm.evaluate --device cuda:0                 # every complete run, both views
    python -m lstm.evaluate --runs lstm/runs/production/s/seed13 --views best_validation
    python -m lstm.evaluate --collect                       # campaign table only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from comparison.blimp_scorer import evaluate_blimp, make_window_batch, neural_window_scorer
from comparison.copy_task import DEFAULT_OUTPUT as COPY_MANIFEST, load_manifest as load_copy_manifest, summarize as summarize_copy
from comparison.protocol import DEFAULT_PROTOCOL, load_protocol
from comparison.same_target import (
    DEFAULT_OUTPUT_ROOT as SAME_TARGET_ROOT, FULL, iter_target_histories, load_manifest as load_same_target_manifest,
    summarize_scores,
)
from dataset.common import DEFAULT_DATA_ROOT, PROJECT_ROOT, file_record, read_json, sha256_file, utc_now, write_json
from dataset.encode import split_arrays
from dataset.model_inputs import NextTokenWindowDataset, create_batch_loader
from dataset.windows import WINDOW_TARGETS, load_windows
from ngram.evaluate import summarize_windows
from .model import LSTMConfig, LSTMLM

VIEWS = {"best_validation": "best.pt", "final_budget": "final.pt"}
VALIDATION_TOLERANCE = 1e-4  # nats; gate before any test scoring
DEFAULT_RESULTS = DEFAULT_DATA_ROOT.parent / "results" / "lstm"
RUN_GLOBS = ("lstm/runs/production/*/seed*", "lstm/runs/sensitivity_*/*/seed*")
CODE_SHA256 = sha256_file(Path(__file__))


def strict_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def run_label(run_dir: Path) -> str:
    """lstm_<campaign>_<size>_seed<N>, e.g. lstm_production_m_deep_seed13."""
    run_dir = Path(run_dir)
    return f"lstm_{run_dir.parents[1].name}_{run_dir.parent.name}_{run_dir.name}"


def discover_runs(root: Path = PROJECT_ROOT) -> list[Path]:
    return sorted(path for pattern in RUN_GLOBS for path in Path(root).glob(pattern) if path.is_dir())


def run_status(run_dir: Path) -> str:
    summary = Path(run_dir) / "summary.json"
    return read_json(summary).get("status", "unknown") if summary.is_file() else "incomplete_or_diverged"


def load_checkpoint(run_dir: Path, view: str, device: torch.device | str) -> tuple[LSTMLM, dict, dict]:
    """Model in eval mode on device, the run identity, and checkpoint metadata."""
    run_dir = Path(run_dir)
    identity = read_json(run_dir / "run.json")["identity"]
    if identity.get("architecture") != "lstm":
        raise ValueError(f"Not an LSTM run: {run_dir}")
    path = run_dir / VIEWS[view]
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("identity") != identity:
        raise ValueError(f"Checkpoint identity differs from run.json: {path}")
    model = LSTMLM(LSTMConfig(**identity["model"]))
    model.load_state_dict(saved["model"])
    model.to(device).eval()
    meta = {"view": view, "file": path.name, "sha256": sha256_file(path),
            "consumed_tokens": int(saved["consumed_tokens"])}
    return model, identity, meta


def token_nll(model: LSTMLM, batch: dict[str, torch.Tensor]) -> np.ndarray:
    """Per-target natural-log NLL [B, T] in float64; NaN where loss_mask is false."""
    logits = model(batch["input_ids"]).float()
    nll = F.cross_entropy(logits.transpose(1, 2), batch["target_ids"], reduction="none").double()
    nll = nll.masked_fill(~batch["loss_mask"], math.nan).cpu().numpy()
    if not np.isfinite(nll[batch["loss_mask"].cpu().numpy()]).all():
        raise ValueError("Non-finite NLL for a scored target")
    return nll


def window_nll(model: LSTMLM, dataset: NextTokenWindowDataset, device, batch_size: int) -> np.ndarray:
    """NLL [windows, 512] in the frozen evaluation order (as the trainer's evaluate)."""
    rows = []
    with torch.inference_mode():
        for raw in create_batch_loader(dataset, batch_size=batch_size, training=False):
            batch = {key: value.to(device) for key, value in raw.items() if torch.is_tensor(value)}
            rows.append(token_nll(model, batch))
    return np.concatenate(rows)


def sequences_nll(model: LSTMLM, sequences: list[list[int]], device, batch_size: int) -> list[np.ndarray]:
    """Per-target NLL for each target sequence, each from [BOS] and fresh state (right padded)."""
    out: list[np.ndarray | None] = [None] * len(sequences)
    order = sorted(range(len(sequences)), key=lambda index: len(sequences[index]))
    with torch.inference_mode():
        for start in range(0, len(order), batch_size):
            chosen = order[start:start + batch_size]
            nll = token_nll(model, make_window_batch([sequences[index] for index in chosen], device))
            for row, index in enumerate(chosen):
                out[index] = nll[row, :len(sequences[index])]
    return out


def publish(output: Path, write: Callable[[Path], dict]) -> dict:
    """Write into a staging folder and rename it into place; never overwrite."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite results: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        result = write(staging)
        staging.rename(output)
    return result


def provenance(identity: dict, checkpoint: dict, label: str) -> dict:
    return {"model_label": label, "checkpoint": checkpoint, "run_identity": identity,
            "scorer_code_sha256": CODE_SHA256, "torch_version": torch.__version__,
            "numerics": "fp32_no_autocast_tf32_disabled_eval_mode_fresh_zero_state_per_input"}


def validation_check(model, identity, checkpoint, run_dir: Path, output: Path, *, token_root: Path,
                     device, batch_size: int) -> dict:
    """Recompute validation NLL and compare with the value the trainer recorded for this checkpoint."""
    if checkpoint["view"] == "best_validation":
        expected = read_json(Path(run_dir) / "summary.json")["best_validation_nll"]
    else:
        events = [json.loads(line) for line in (Path(run_dir) / "metrics.jsonl").open(encoding="utf-8")]
        final = [row for row in events if row.get("event") == "validation"
                 and row["consumed_tokens"] == checkpoint["consumed_tokens"]]
        expected = final[-1]["nll"]
    nll = window_nll(model, NextTokenWindowDataset(token_root, "validation"), device, batch_size)
    recomputed = float(np.nansum(nll) / np.isfinite(nll).sum())
    result = {"schema_version": 1, "created_at": utc_now(), "expected_nll": expected,
              "recomputed_nll": recomputed, "abs_difference": abs(recomputed - expected),
              "tolerance": VALIDATION_TOLERANCE, "targets": int(np.isfinite(nll).sum()),
              "status": "PASS" if abs(recomputed - expected) <= VALIDATION_TOLERANCE else "FAIL",
              **provenance(identity, checkpoint, output.parent.parent.name)}
    write_json(output, result)
    return result


def evaluate_test(model, identity, checkpoint, label, output: Path, *, token_root: Path, device,
                  batch_size: int, protocol: Path = DEFAULT_PROTOCOL) -> dict:
    window_root = token_root / "windows"
    load_windows(window_root, token_root)  # Verifies hashes and coverage first.
    index = np.load(window_root / "test.windows.npy", allow_pickle=False)
    nll = window_nll(model, NextTokenWindowDataset(token_root, "test"), device, batch_size)
    if nll.shape != (len(index), WINDOW_TARGETS) or np.isfinite(nll).sum() != index[:, 2].sum():
        raise ValueError("Test NLL does not cover exactly the valid targets")
    summary = summarize_windows(nll, np.zeros(len(index), dtype=np.int64), index,
                                load_protocol(protocol)["evaluation"]["context"])

    def write(staging: Path) -> dict:
        np.savez_compressed(staging / "windows.npz", index=index, nll=nll)
        result = {"schema_version": 1, "status": "PASS", "created_at": utc_now(), "split": "test",
                  "protocol_sha256": sha256_file(protocol),
                  "token_manifest_sha256": sha256_file(token_root / "manifest.json"),
                  "window_manifest_sha256": sha256_file(window_root / "manifest.json"),
                  **provenance(identity, checkpoint, label),
                  "windows_file": file_record(staging / "windows.npz", staging), **summary}
        write_json(staging / "summary.json", result)
        return result

    return publish(output, write)


def score_same_target(model, targets: np.ndarray, tokens, offsets, rules, *, device, batch_size: int,
                      check_rows: int = 256) -> tuple[dict[str, np.ndarray], dict]:
    """NLL of each manifest target under every history; input [BOS] + history, last logit only.

    Full history uses one causal forward per window (the protocol's allowed equivalent);
    the first `check_rows` targets are also scored with explicit [BOS] + prefix inputs.
    """
    nll = {condition: np.empty(len(targets), dtype=np.float64) for condition in rules.conditions}
    # Full: one forward per window, then read the late positions of that window.
    windows = sorted({(int(a), int(s)) for a, s, _, _ in targets})
    window_ids = [np.asarray(tokens[int(offsets[a]) + s:][:WINDOW_TARGETS], dtype=np.int64).tolist()
                  for a, s in windows]
    per_window = dict(zip(windows, sequences_nll(model, window_ids, device, batch_size), strict=True))
    for row, (article, start, position, _) in enumerate(targets):
        nll[FULL][row] = per_window[int(article), int(start)][int(position) - 1]
    # Restricted histories: one fresh input per target and condition, batched.
    pending: dict[str, list[tuple[int, list[int]]]] = {c: [] for c in rules.conditions if c != FULL}
    explicit_full: list[list[int]] = []

    def flush(condition: str) -> None:
        rows, sequences = zip(*pending[condition])
        for row, values in zip(rows, sequences_nll(model, list(sequences), device, batch_size), strict=True):
            nll[condition][row] = values[-1]
        pending[condition].clear()

    for row, target, histories in iter_target_histories(targets, tokens, offsets, rules):
        for condition, history in histories.items():
            if condition == FULL:
                if row < check_rows:
                    explicit_full.append(history.tolist() + [target])
                continue
            pending[condition].append((row, history.tolist() + [target]))
            if len(pending[condition]) == batch_size:
                flush(condition)
    for condition in pending:
        if pending[condition]:
            flush(condition)
    explicit = np.array([values[-1] for values in sequences_nll(model, explicit_full, device, batch_size)])
    deviation = float(np.abs(explicit - nll[FULL][:len(explicit)]).max())
    check = {"rows": len(explicit), "max_abs_deviation": deviation,
             "status": "PASS" if deviation <= VALIDATION_TOLERANCE else "FAIL"}
    return nll, check


def evaluate_same_target(model, identity, checkpoint, label, output: Path, *, manifest_root: Path,
                         token_root: Path, device, batch_size: int, protocol: Path = DEFAULT_PROTOCOL) -> dict:
    manifest, targets, rules = load_same_target_manifest(manifest_root, protocol)
    tokens, offsets = split_arrays(token_root, manifest["split"])
    nll, check = score_same_target(model, targets, tokens, offsets, rules, device=device, batch_size=batch_size)
    if check["status"] != "PASS":
        raise ValueError(f"Full-history shortcut differs from explicit prefixes: {check}")

    def write(staging: Path) -> dict:
        np.savez_compressed(staging / "per_target_nll.npz", targets=targets, **nll)
        result = {"schema_version": 1, "status": "PASS", "created_at": utc_now(), "split": manifest["split"],
                  "coverage": manifest["coverage"], "start_symbol": "bos",
                  "full_history_equivalence_check": check, **summarize_scores(nll, rules.conditions),
                  "manifest_sha256": sha256_file(Path(manifest_root) / "manifest.json"),
                  "manifest_provenance": manifest["provenance"], **provenance(identity, checkpoint, label),
                  "scores_file": file_record(staging / "per_target_nll.npz", staging)}
        write_json(staging / "summary.json", result)
        return result

    return publish(output, write)


def evaluate_copy(model, identity, checkpoint, label, output: Path, *, manifest_root: Path, device,
                  batch_size: int, protocol: Path = DEFAULT_PROTOCOL) -> dict:
    manifest, spans = load_copy_manifest(manifest_root, protocol)
    results, arrays = {}, {}
    for (source, length), span in sorted(spans.items()):
        a, b = span["a"].astype(np.int64), span["b"].astype(np.int64)
        nll = {"repeat": np.stack(sequences_nll(model, np.concatenate([a, a], 1).tolist(), device, batch_size)),
               "control": np.stack(sequences_nll(model, np.concatenate([a, b], 1).tolist(), device, batch_size))}
        results.setdefault(source, {})[str(length)] = summarize_copy(nll)
        arrays.update({f"{source}_L{length}_{condition}": values for condition, values in nll.items()})
    primary_length = manifest["primary_length"]

    def write(staging: Path) -> dict:
        np.savez_compressed(staging / "per_target_nll.npz", **arrays)
        result = {"schema_version": 1, "status": "PASS", "created_at": utc_now(), "split": manifest["split"],
                  "start_symbol": "bos",
                  f"natural_copy_{primary_length}":
                      results["test_natural_text"][str(primary_length)]["all_positions"]["copy_gain"],
                  "results": results, "manifest_sha256": sha256_file(Path(manifest_root) / "manifest.json"),
                  "manifest_provenance": manifest["provenance"], **provenance(identity, checkpoint, label),
                  "scores_file": file_record(staging / "per_target_nll.npz", staging)}
        write_json(staging / "summary.json", result)
        return result

    return publish(output, write)


def evaluate_view(run_dir: Path, view: str, results: Path, *, token_root: Path, device, batch_size: int) -> dict:
    """Validation gate, then every test-side evaluation that is not finished yet."""
    label = run_label(run_dir)
    out = Path(results) / label / view
    model, identity, checkpoint = load_checkpoint(run_dir, view, device)
    gate_file = out / "validation_check.json"
    gate = read_json(gate_file) if gate_file.is_file() else validation_check(
        model, identity, checkpoint, run_dir, gate_file, token_root=token_root, device=device, batch_size=batch_size)
    print(label, view, "validation", gate["status"], f"diff={gate['abs_difference']:.2e}", flush=True)
    if gate["status"] != "PASS":
        raise ValueError(f"Validation gate failed for {label} {view}: {gate_file}")
    done = {}
    if not (out / "test").exists():
        done["test"] = evaluate_test(model, identity, checkpoint, label, out / "test", token_root=token_root,
                                     device=device, batch_size=batch_size)["perplexity"]
    if not (out / "blimp").exists():
        scorer = neural_window_scorer(lambda batch: model(batch["input_ids"]), device=device, batch_size=4 * batch_size)
        done["blimp"] = evaluate_blimp(scorer, out / "blimp", model_label=f"{label}_{view}")["primary"]["overall"]["accuracy"]
    if not (out / "same_target").exists():
        done["same_target"] = evaluate_same_target(
            model, identity, checkpoint, label, out / "same_target", manifest_root=SAME_TARGET_ROOT / "manifest_test",
            token_root=token_root, device=device, batch_size=batch_size)["truncation_gain"]
    if not (out / "copy").exists():
        done["copy"] = evaluate_copy(model, identity, checkpoint, label, out / "copy", manifest_root=COPY_MANIFEST,
                                     device=device, batch_size=batch_size)["natural_copy_100"]
    print(label, view, done or "already complete", flush=True)
    return done


def collect(results: Path, runs: list[Path]) -> dict:
    """One table of headline numbers per run and view, plus runs that were not scored."""
    rows, unscored = [], []
    for run_dir in runs:
        label = run_label(run_dir)
        if run_status(run_dir) != "complete":
            unscored.append({"label": label, "status": run_status(run_dir)})
            continue
        for view in VIEWS:
            out = Path(results) / label / view

            def get(part: str) -> dict | None:
                path = out / part / "summary.json"
                return read_json(path) if path.is_file() else None

            test, blimp, same, copy = get("test"), get("blimp"), get("same_target"), get("copy")
            rows.append({
                "label": label, "view": view,
                "test_perplexity": test and test["perplexity"],
                "test_context_gain": test and test["context"]["context_gain"],
                "blimp_accuracy": blimp and blimp["primary"]["overall"]["accuracy"],
                "same_target_truncation_gain": same and same["truncation_gain"],
                "same_target_shuffle_gain": same and same["shuffle_gain"],
                "natural_copy_100": copy and copy["natural_copy_100"],
            })
    table = {"schema_version": 1, "created_at": utc_now(), "rows": rows, "unscored_runs": unscored}
    write_json(Path(results) / "campaign_summary.json", table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, nargs="*", help="Default: every production and sensitivity run")
    parser.add_argument("--views", nargs="*", choices=list(VIEWS), default=list(VIEWS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--collect", action="store_true", help="Only rebuild campaign_summary.json")
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    strict_fp32()
    runs = [path.resolve() for path in args.runs] if args.runs else discover_runs()
    if not args.collect:
        for run_dir in runs:
            if run_status(run_dir) != "complete":
                print(run_label(run_dir), "not scored:", run_status(run_dir), flush=True)
                continue
            for view in args.views:
                evaluate_view(run_dir, view, args.results, token_root=args.tokens, device=torch.device(args.device),
                              batch_size=args.batch_size)
    table = collect(args.results, runs)
    print({"rows": len(table["rows"]), "unscored": table["unscored_runs"]})


if __name__ == "__main__":
    main()
