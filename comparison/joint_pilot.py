"""Build the joint Transformer/LSTM pilot report that unlocks the production snapshot.

The Transformer side comes from the Transformer owner's report (transformer/timing_pilot.py,
status TRANSFORMER_ONLY). The LSTM side is measured here from finished LSTM pilot runs with the
SAME method: prediction tokens / trainer step seconds over full updates, excluding the first
warm-up updates and the final budget update. The joint report is bound to the CURRENT protocol and
input manifests (what `comparison.protocol` checks). The Transformer report's own protocol hash is
kept as provenance; the protocol changes after it (LSTM embedding init, tuning-grid extension) do
not touch Transformer timing, so it is reused instead of re-measured.

    python -m comparison.joint_pilot --transformer-report transformer_pilot.json \
        --lstm-runs lstm/runs/pilot_mb16_5M --output comparison/pilots/joint_pilot.json \
        --exact-tokens 250000000 --tuning-tokens 50000000
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
from pathlib import Path

import torch

from comparison.protocol import DEFAULT_PROTOCOL, apply_pilot, load_protocol
from dataset.common import DEFAULT_DATA_ROOT, read_json, sha256_file, utc_now, write_json

SIZES = ("s", "m", "l")


def summarize_updates(events: list[dict], *, warmup_updates: int, effective_batch: int, budget: int) -> dict:
    """Same rule as the Transformer pilot: weight by real targets; skip warm-up and partial/final updates."""
    previous_seconds, previous_tokens, tokens_sum, seconds_sum, rates = 0.0, 0, 0, 0.0, []
    updates = [event for event in events if event["event"] == "update"]
    for index, event in enumerate(updates, start=1):
        seconds = event["train_seconds"] - previous_seconds
        tokens = event["consumed_tokens"] - previous_tokens
        if event["updates"] != index or tokens != event["prediction_tokens"] or seconds <= 0:
            raise ValueError("Expected contiguous, unresumed update counters")
        if not all(math.isfinite(event[key]) for key in ("training_nll", "gradient_norm", "learning_rate")):
            raise ValueError("Nonfinite update metric")
        if index > warmup_updates and event["batch_windows"] == effective_batch and event["consumed_tokens"] < budget:
            tokens_sum, seconds_sum = tokens_sum + tokens, seconds_sum + seconds
            rates.append(tokens / seconds)
        previous_seconds, previous_tokens = event["train_seconds"], event["consumed_tokens"]
    if previous_tokens != budget or len(rates) < 20:
        raise ValueError("Pilot run incomplete or too short for a throughput measurement")
    return {"prediction_tokens_per_second": tokens_sum / seconds_sum, "measured_prediction_tokens": tokens_sum,
            "measured_training_seconds": seconds_sum, "measured_updates": len(rates),
            "warmup_updates_excluded": warmup_updates,
            "update_tokens_per_second_median": statistics.median(rates)}


def lstm_measurement(run_dir: Path, *, microbatch: int, effective_batch: int, warmup_updates: int) -> dict:
    run, summary = read_json(run_dir / "run.json"), read_json(run_dir / "summary.json")
    identity = run.get("identity", run)
    if (summary["status"] != "complete" or summary.get("architecture") != "lstm"
            or identity["microbatch_windows"] != microbatch):
        raise ValueError(f"Not a complete LSTM pilot run at microbatch {microbatch}: {run_dir}")
    if identity["protocol_sha256"] != sha256_file(DEFAULT_PROTOCOL):
        raise ValueError(f"LSTM pilot ran under another protocol: {run_dir}")
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    validations = [event for event in events if event["event"] == "validation"]
    checkpoint_bytes = (run_dir / "latest.pt").stat().st_size
    return {"architecture": "lstm", "size": summary["size"],
            **summarize_updates(events, warmup_updates=warmup_updates, effective_batch=effective_batch,
                                budget=identity["budget_tokens"]),
            "peak_memory_bytes": summary["peak_cuda_bytes"], "validation_seconds": validations[-1]["seconds"],
            "checkpoint_bytes": checkpoint_bytes, "final_validation_nll": validations[-1]["nll"],
            "learning_rate": identity["learning_rate"], "dropout": identity.get("dropout"),
            "pilot_prediction_tokens": identity["budget_tokens"], "run_dir": str(run_dir)}


def hardware_record() -> dict:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return {"host": platform.node(), "gpus": sorted(set(gpu)), "python": platform.python_version(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda}


def build(args: argparse.Namespace) -> dict:
    protocol = load_protocol()
    effective_batch = protocol["training"]["effective_batch_windows"]
    transformer = read_json(args.transformer_report)
    if transformer["settings"]["microbatch_windows"] != args.microbatch:
        raise ValueError("Transformer pilot used another microbatch")
    # Manifest FILES embed created_at, so their hashes differ per machine even for identical data.
    # Bind the report to this machine's manifests (what the snapshot checks) and require the same
    # tokenizer; identical token/window CONTENT across machines is established by the reference
    # hashes in dataset/expected_outputs.lock.json (status confirmed by both authors).
    token_manifest = sha256_file(args.token_dir / "manifest.json")
    window_manifest = sha256_file(args.token_dir / "windows/manifest.json")
    if transformer["tokenizer_sha256"] != protocol["inputs"]["tokenizer_sha256"]:
        raise ValueError("Transformer pilot used another tokenizer")
    reference = read_json(Path(__file__).resolve().parents[1] / "dataset/expected_outputs.lock.json")
    if reference.get("status") != "confirmed":
        raise ValueError("Cross-machine data identity is not confirmed")
    rows = [dict(row, architecture="transformer") for row in transformer["measurements"]]
    warmup = transformer.get("pilot_settings", {}).get("warmup_updates", 10)
    rows += [lstm_measurement(args.lstm_runs / size, microbatch=args.microbatch, effective_batch=effective_batch,
                              warmup_updates=warmup) for size in SIZES]
    report = {
        "schema_version": 1, "status": "PASS", "created_at": utc_now(),
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL),
        "token_manifest_sha256": token_manifest, "window_manifest_sha256": window_manifest,
        "hardware": {"transformer": transformer["hardware"], "lstm": hardware_record(),
                     "note": "Architectures were measured on different GPUs; throughput is not comparable "
                             "across architectures. Timing does not affect results (float32, same trainer)."},
        "settings": {"exact_prediction_tokens": args.exact_tokens, "tuning_prediction_tokens": args.tuning_tokens,
                     "microbatch_windows": args.microbatch,
                     "gradient_accumulation_steps": effective_batch // args.microbatch},
        "measurements": rows,
        "provenance": {
            "transformer_report_sha256": sha256_file(args.transformer_report),
            "transformer_report_protocol_sha256": transformer["protocol_sha256"],
            "transformer_report_token_manifest_sha256": transformer["token_manifest_sha256"],
            "transformer_report_window_manifest_sha256": transformer["window_manifest_sha256"],
            "cross_machine_data_identity": "dataset/expected_outputs.lock.json (confirmed reference hashes)",
            "protocol_changes_since_transformer_report": [
                "training.initialization.lstm_token_embedding (LSTM only)",
                "tuning.learning_rates extended to 1e-2 and 3e-2 (agreed; timing-neutral)"],
        },
    }
    apply_pilot(protocol, report)  # Fail now, not at snapshot time, if anything is missing.
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transformer-report", type=Path, required=True)
    parser.add_argument("--lstm-runs", type=Path, required=True, help="folder with s/, m/, l/ LSTM pilot runs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-dir", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--exact-tokens", type=int, required=True)
    parser.add_argument("--tuning-tokens", type=int, required=True)
    parser.add_argument("--microbatch", type=int, default=16)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite {args.output}")
    report = build(args)
    write_json(args.output, report)
    for row in report["measurements"]:
        print(f"{row['architecture']:>11} {row['size']}: {row['prediction_tokens_per_second']:>10,.0f} tok/s  "
              f"peak {row['peak_memory_bytes'] / 2**30:.2f} GiB  val {row['validation_seconds']:.2f}s  "
              f"ckpt {row['checkpoint_bytes'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
