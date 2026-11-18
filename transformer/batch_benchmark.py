"""Compare microbatch throughput with the unchanged float32 trainer on Slurm.

These short seed-101 runs are timing evidence, never final research runs.
Keep 64 windows per update, the selected S hyperparameters, and all data rules.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import torch

from comparison.protocol import check_lock, freeze
from dataset.common import PROJECT_ROOT, read_json, sha256_file, utc_now, write_json
from transformer import timing_pilot, tuning


def code_hashes() -> dict[str, str]:
    names = ("transformer/batch_benchmark.py", "transformer/batch_benchmark.sbatch",
             "transformer/timing_pilot.py", "transformer/train.py", "transformer/model.py",
             "dataset/model_inputs.py")
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def prepare(root: Path, tuning_root: Path, budget: int = 2_000_000) -> dict:
    root, tuning_root = root.resolve(), tuning_root.resolve()
    if type(budget) is not int or budget < 1_100_000:
        raise ValueError("Use at least 1.1M pilot tokens for 10 warm-up and 20 measured updates")
    selected = tuning.collect(tuning_root)
    if selected["status"] != "finished" or selected["failed_trials"] or selected["winner"] is None:
        raise ValueError("Finish all tuning trials successfully before benchmarking the winner")
    study = tuning.verify_study(tuning_root)
    settings = {"schema_version": 1, "purpose": "transformer_microbatch_timing_only",
                "budget_tokens": budget, "seed": study["seed"],
                "learning_rate": selected["winner"]["learning_rate"],
                "dropout": selected["winner"]["dropout"],
                "token_dir": study["token_dir"], "tokenizer_dir": study["tokenizer_dir"],
                "tuning_study_sha256": sha256_file(tuning_root / "study.json"),
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "code_sha256": code_hashes(),
                "tasks": [{"index": i, "size": size, "microbatch_windows": batch,
                           "directory": f"{size}_mb{batch}"}
                          for i, (size, batch) in enumerate(
                              (size, batch) for size in ("l", "m", "s") for batch in (64, 32, 16))]}
    if root.exists():
        previous = verify(root)
        if any(previous.get(key) != value for key, value in settings.items()):
            raise ValueError("Existing benchmark settings differ; choose another directory")
        return previous
    root.mkdir(parents=True)
    freeze(Path(study["token_dir"]), artifacts=Path(study["tokenizer_dir"]),
           output=root / "protocol.prepared.lock.json")
    settings.update(created_at=utc_now(),
                    snapshot_sha256=sha256_file(root / "protocol.prepared.lock.json"))
    write_json(root / "benchmark.json", settings)
    return settings


def verify(root: Path) -> dict:
    settings = read_json(root / "benchmark.json")
    if (settings["code_sha256"] != code_hashes()
            or settings["torch_version"] != torch.__version__
            or settings["cuda_version"] != torch.version.cuda):
        raise ValueError("Benchmark code/runtime changed; retain evidence and prepare a new directory")
    if settings["snapshot_sha256"] != sha256_file(root / "protocol.prepared.lock.json"):
        raise ValueError("Benchmark snapshot changed")
    check_lock(root / "protocol.prepared.lock.json", Path(settings["token_dir"]),
               artifacts=Path(settings["tokenizer_dir"]))
    return settings


def result_for(root: Path, task: dict) -> dict | None:
    attempts = sorted((root / task["directory"]).glob("attempt_*/transformer_pilot.json"))
    if not attempts:
        return None
    report = read_json(attempts[-1])
    report["report_path"] = str(attempts[-1])
    return report


def run_task(root: Path, index: int) -> None:
    settings = verify(root)
    if not 0 <= index < len(settings["tasks"]):
        raise ValueError("Benchmark task index is outside the frozen grid")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Run this benchmark inside a Slurm GPU allocation")
    task = settings["tasks"][index]
    with (root / f"task_{index:02d}.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = result_for(root, task)
        if previous and previous["status"] == "TRANSFORMER_ONLY":
            print(f"Already complete: {task['directory']}", flush=True)
            return
        output = root / task["directory"] / f"attempt_{time.time_ns()}"
        args = argparse.Namespace(
            output_dir=output, sizes=[task["size"]], token_dir=Path(settings["token_dir"]),
            tokenizer_dir=Path(settings["tokenizer_dir"]), budget_tokens=settings["budget_tokens"],
            microbatch_windows=task["microbatch_windows"], seed=settings["seed"],
            learning_rate=settings["learning_rate"], dropout=settings["dropout"],
            warmup_updates=10, minimum_measured_updates=20)
        timing_pilot.run_pilot(args)


def recommend(rows: list[dict], batches: tuple[int, ...] = (16, 32, 64)) -> dict:
    """Choose one shared microbatch by aggregate wall time, with 10% VRAM headroom.

    A large allocation alone is not a speed objective. Wall time includes input
    setup, validation, logging and checkpoints; loss values never enter selection.
    """
    candidates = []
    for batch in batches:
        group = [row for row in rows if row["microbatch_windows"] == batch]
        if (len(group) != 3 or {row["size"] for row in group} != {"s", "m", "l"}
                or any(row["status"] != "complete" or not row["memory_headroom_ok"] for row in group)):
            continue
        seconds = sum(row["wall_seconds_per_million_tokens"] for row in group)
        candidates.append({"microbatch_windows": batch, "gradient_accumulation_steps": 64 // batch,
                           "sum_wall_seconds_per_million_tokens": seconds})
    return min(candidates, key=lambda row: (row["sum_wall_seconds_per_million_tokens"],
                                           row["microbatch_windows"])) if candidates else {}


def collect(root: Path) -> dict:
    settings = verify(root)
    snapshot = read_json(root / "protocol.prepared.lock.json")
    rows = []
    for task in settings["tasks"]:
        report = result_for(root, task)
        row = {**task, "status": "pending" if report is None else report["status"]}
        if report and report["status"] == "TRANSFORMER_ONLY":
            if (len(report["measurements"]) != 1
                    or any(report[key] != snapshot[key] for key in (
                        "protocol_sha256", "token_manifest_sha256", "window_manifest_sha256"))
                    or report["train_code_sha256"] != settings["code_sha256"]["transformer/train.py"]
                    or report["model_code_sha256"] != settings["code_sha256"]["transformer/model.py"]
                    or report["settings"]["microbatch_windows"] != task["microbatch_windows"]
                    or report["pilot_settings"]["dropout"] != settings["dropout"]
                    or report["pilot_settings"]["learning_rate"] != settings["learning_rate"]):
                raise ValueError("Benchmark report provenance/settings differ")
            measured = report["measurements"][0]
            if measured["size"] != task["size"]:
                raise ValueError("Benchmark size differs")
            capacity = report["hardware"]["gpu_total_memory_bytes"]
            row.update(status="complete", report_path=report["report_path"],
                       prediction_tokens_per_second=measured["prediction_tokens_per_second"],
                       wall_seconds_per_million_tokens=(measured["subprocess_wall_seconds"]
                                                       * 1_000_000 / settings["budget_tokens"]),
                       peak_memory_bytes=measured["peak_memory_bytes"], gpu_memory_bytes=capacity,
                       memory_headroom_ok=measured["peak_memory_bytes"] <= 0.90 * capacity)
        rows.append(row)
    finished = all(row["status"] in ("complete", "FAILED") for row in rows)
    return {"status": "finished" if finished else "incomplete", "production_ready": False,
            "benchmark_dir": str(root), "benchmark_sha256": sha256_file(root / "benchmark.json"),
            "rows": rows, "recommendation": recommend(rows) if finished else {},
            "limitations": "One short timing run per candidate on two shared-server GPUs. "
                           "Fastest observed common batch, not a guarantee of maximal throughput. "
                           "Apply only after agreeing the shared pilot settings with Hugo."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    preparing = sub.add_parser("prepare")
    preparing.add_argument("--tuning-study", type=Path, required=True)
    preparing.add_argument("--budget-tokens", type=int, default=2_000_000)
    for name in ("run", "collect"):
        command = sub.add_parser(name)
        if name == "run":
            command.add_argument("--task-index", type=int, required=True)
    for command in (preparing, *[sub.choices[name] for name in ("run", "collect")]):
        command.add_argument("--benchmark-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.benchmark_dir.resolve()
    if args.action == "prepare":
        result = prepare(root, args.tuning_study, args.budget_tokens)
    elif args.action == "run":
        run_task(root, args.task_index)
        return
    else:
        result = collect(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
