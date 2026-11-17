"""Measure Transformer S/M/L with the existing, unchanged exploratory trainer.

This produces the Transformer contribution to a joint pilot, not a production
lock. Throughput uses the trainer's cumulative step timer after warm-up; it is
not a separate CUDA-event benchmark or an end-to-end wall-clock rate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from comparison.protocol import freeze
from dataset.common import DEFAULT_DATA_ROOT, PROJECT_ROOT, read_json, sha256_file, utc_now, write_json
from tokenizer.core import DEFAULT_OUTPUT as DEFAULT_TOKENIZER_DIR


def positive(value: object, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"Missing/nonpositive/nonfinite measurement: {name}")
    return float(value)


def summarize_updates(events: list[dict], *, warmup_updates: int, minimum_updates: int,
                      effective_batch: int, budget: int) -> dict:
    """Weight by actual nonpadding tokens; omit warm-up and partial/final updates."""
    previous_seconds, previous_tokens = 0.0, 0
    measured_tokens, measured_seconds = 0, 0.0
    rates = []
    updates = [event for event in events if event["event"] == "update"]
    for index, event in enumerate(updates, start=1):
        seconds = positive(event["train_seconds"] - previous_seconds, "update seconds")
        tokens = event["consumed_tokens"] - previous_tokens
        if event["updates"] != index or tokens != event["prediction_tokens"]:
            raise ValueError("Expected contiguous, unresumed update/token counters")
        positive(tokens, "update prediction tokens")
        for key in ("training_nll", "gradient_norm", "learning_rate"):
            if not math.isfinite(event[key]):
                raise ValueError(f"Nonfinite update metric: {key}")
        if (index > warmup_updates and event["batch_windows"] == effective_batch
                and event["consumed_tokens"] < budget):
            measured_tokens += tokens
            measured_seconds += seconds
            rates.append(tokens / seconds)
        previous_seconds = event["train_seconds"]
        previous_tokens = event["consumed_tokens"]
    if previous_tokens != budget:
        raise ValueError("Training did not reach the exact pilot token budget")
    if len(rates) < minimum_updates:
        raise ValueError(f"Only {len(rates)} measured full updates; need {minimum_updates}. "
                         "Increase --budget-tokens; retain this run as evidence.")
    return {
        "prediction_tokens_per_second": measured_tokens / measured_seconds,
        "measured_prediction_tokens": measured_tokens,
        "measured_training_seconds": measured_seconds,
        "measured_updates": len(rates),
        "warmup_updates_excluded": warmup_updates,
        "update_tokens_per_second_min": min(rates),
        "update_tokens_per_second_median": statistics.median(rates),
        "update_tokens_per_second_max": max(rates),
    }


def hardware_record() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("A visible allocated CUDA GPU is required; CPU timings cannot enter this pilot")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,pci.bus_id", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        driver = None
    return {
        "hostname": platform.node(), "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unavailable")),
        "gpu_total_memory_bytes": properties.total_memory,
        "compute_capability": [properties.major, properties.minor],
        "logical_cuda_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "python_version": platform.python_version(), "python_executable": sys.executable,
        "cpu_threads": torch.get_num_threads(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "nvidia_smi_inventory": driver,
    }


def collect_measurement(run_dir: Path, *, args: argparse.Namespace, size: str,
                        expected_identity: dict, wall_seconds: float) -> dict:
    summary = read_json(run_dir / "summary.json")
    identity = read_json(run_dir / "run.json")["identity"]
    expected = {
        **expected_identity, "size": size, "mode": "exploratory",
        "budget_tokens": args.budget_tokens, "microbatch_windows": args.microbatch_windows,
        "effective_batch_windows": 64, "seed": args.seed,
        "learning_rate": args.learning_rate,
    }
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Trainer identity differs from pilot inputs/settings/code")
    if identity["model"]["dropout"] != args.dropout:
        raise ValueError("Trainer dropout differs from pilot settings")
    if (summary["status"] != "complete" or summary["mode"] != "exploratory"
            or summary["consumed_tokens"] != args.budget_tokens):
        raise ValueError("Only complete exploratory pilot runs can be exported")
    events = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    timing = summarize_updates(events, warmup_updates=args.warmup_updates,
                               minimum_updates=args.minimum_measured_updates,
                               effective_batch=64, budget=args.budget_tokens)
    validations = [event for event in events if event["event"] == "validation"]
    if (not validations or validations[0]["consumed_tokens"] != 0
            or validations[-1]["consumed_tokens"] != args.budget_tokens):
        raise ValueError("Missing initial/final full validation")
    for event in validations:
        positive(event["tokens"], "validation targets")
        positive(event["seconds"], "validation seconds")
        positive(event["nll"], "validation NLL")
    if len({event["tokens"] for event in validations}) != 1:
        raise ValueError("Validation target population changed during pilot")
    checkpoint_bytes = (run_dir / "latest.pt").stat().st_size
    if checkpoint_bytes != summary["latest_checkpoint_bytes"]:
        raise ValueError("Checkpoint size differs from completed summary")
    positive(summary["peak_cuda_bytes"], "peak CUDA allocation")
    return {
        "architecture": "transformer", "size": size,
        **timing,
        "peak_memory_bytes": summary["peak_cuda_bytes"],
        "validation_seconds": validations[-1]["seconds"],
        "checkpoint_bytes": checkpoint_bytes,
        "validation_nll_start": validations[0]["nll"],
        "validation_nll_final": validations[-1]["nll"],
        "validation_prediction_tokens": validations[-1]["tokens"],
        "trainer_prediction_tokens_per_second_including_warmup": summary["prediction_tokens_per_second"],
        "subprocess_wall_seconds": wall_seconds,
        "parameters": summary["parameters"],
        "run_directory": str(run_dir),
        "artifact_sha256": {name: sha256_file(run_dir / name)
                            for name in ("run.json", "summary.json", "metrics.jsonl")},
    }


def run_pilot(args: argparse.Namespace) -> dict:
    if (args.microbatch_windows <= 0 or 64 % args.microbatch_windows
            or args.warmup_updates < 0 or args.minimum_measured_updates < 1
            or args.budget_tokens <= 0 or args.seed < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.dropout) or not 0 <= args.dropout < 1
            or len(set(args.sizes)) != len(args.sizes)):
        raise ValueError("Invalid pilot settings; microbatch must divide 64 and sizes must be unique")
    args.token_dir, args.tokenizer_dir = args.token_dir.resolve(), args.tokenizer_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite pilot directory: {args.output_dir}")
    hardware = hardware_record()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # A prior preparation lock can predate protocol amendments. Freeze current
    # verified inputs locally; retain all older snapshots with their original runs.
    snapshot_path = args.output_dir / "protocol.prepared.lock.json"
    lock = freeze(args.token_dir, artifacts=args.tokenizer_dir, output=snapshot_path)
    config = lock["configuration"]
    if config["training"]["effective_batch_windows"] != 64 or config["training"]["precision"] != "float32":
        raise ValueError("This pilot requires the shared float32 / 64-window protocol")
    expected_identity = {
        key: lock[key] for key in ("protocol_sha256", "token_manifest_sha256", "window_manifest_sha256")
    }
    expected_identity.update({
        "tokenizer_sha256": sha256_file(args.tokenizer_dir / "tokenizer.json"),
        "model_code_sha256": sha256_file(Path(__file__).with_name("model.py")),
        "train_code_sha256": sha256_file(Path(__file__).with_name("train.py")),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    })
    report = {
        "schema_version": 1, "status": "IN_PROGRESS", "production_ready": False,
        "created_at": utc_now(), "hardware": hardware,
        **expected_identity,
        "pilot_script_sha256": sha256_file(Path(__file__)),
        "prepared_snapshot_sha256": sha256_file(snapshot_path),
        "settings": {"microbatch_windows": args.microbatch_windows,
                     "gradient_accumulation_steps": 64 // args.microbatch_windows},
        "pilot_settings": {"prediction_tokens_per_size": args.budget_tokens,
                           "seed": args.seed, "learning_rate": args.learning_rate,
                           "dropout": args.dropout, "precision": "float32",
                           "sizes": args.sizes, "warmup_updates": args.warmup_updates,
                           "minimum_measured_updates": args.minimum_measured_updates},
        "timing_method": "Sum valid targets / sum trainer step-time deltas after warm-up; "
                         "exclude epoch-partial and final updates, validation, checkpoint I/O and logging. "
                         "Uses unchanged trainer timers, not separately synchronized CUDA events. "
                         "Subprocess wall time includes setup, input verification, validation and saves.",
        "limitations": "Transformer-only evidence. Not a production lock, tuning result, or "
                       "proof of equal converged NLL at microbatch 2 and 16. "
                       "Peak memory includes warm-up and validation; allocated bytes, not reserved/total GPU use.",
        "measurements": [],
    }
    report_path = args.output_dir / "transformer_pilot.json"
    write_json(report_path, report)
    try:
        for size in args.sizes:
            run_dir = args.output_dir / size
            command = [
                sys.executable, "-u", "-m", "transformer.train", "--mode", "exploratory",
                "--size", size, "--device", "cuda", "--token-dir", str(args.token_dir),
                "--tokenizer-dir", str(args.tokenizer_dir), "--budget-tokens", str(args.budget_tokens),
                "--microbatch-windows", str(args.microbatch_windows),
                "--learning-rate", str(args.learning_rate), "--dropout", str(args.dropout),
                "--seed", str(args.seed), "--save-every-updates", "25", "--run-dir", str(run_dir),
            ]
            write_json(args.output_dir / f"{size}_command.json", {"argv": command, "cwd": str(PROJECT_ROOT)})
            print(f"Starting Transformer {size.upper()}; log: {args.output_dir / (size + '.log')}", flush=True)
            started = time.perf_counter()
            with (args.output_dir / f"{size}.log").open("x", encoding="utf-8") as log:
                subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            row = collect_measurement(run_dir, args=args, size=size,
                                      expected_identity=expected_identity,
                                      wall_seconds=time.perf_counter() - started)
            report["measurements"].append(row)
            write_json(report_path, report)
            print(f"{size.upper()}: {row['prediction_tokens_per_second']:,.0f} prediction tok/s; "
                  f"peak {row['peak_memory_bytes'] / 2**30:.2f} GiB; "
                  f"validation {row['validation_seconds']:.3f}s", flush=True)
    except Exception as error:
        report.update(status="FAILED", error=str(error))
        write_json(report_path, report)
        raise
    report.update(status="TRANSFORMER_ONLY", completed_at=utc_now())
    write_json(report_path, report)
    print(f"Transformer evidence: {report_path}\nMerge with LSTM S/M/L evidence before production.", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; never overwritten")
    parser.add_argument("--sizes", nargs="+", choices=("s", "m", "l"), default=["s", "m", "l"])
    parser.add_argument("--token-dir", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--budget-tokens", type=int, default=5_000_000, help="Pilot tokens per size, not production/tuning budget")
    parser.add_argument("--microbatch-windows", type=int, default=16)
    parser.add_argument("--warmup-updates", type=int, default=10)
    parser.add_argument("--minimum-measured-updates", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    try:
        run_pilot(args)
    except (ValueError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"Pilot stopped: {error}\n")


if __name__ == "__main__":
    main()
