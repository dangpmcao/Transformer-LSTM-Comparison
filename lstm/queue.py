"""Polite job queue for LSTM runs on a shared multi-GPU server.

Runs the shared trainer (python -m transformer.train --architecture lstm) for a
plan of jobs, using at most --max-gpus GPUs at once and only GPUs that look idle
(low utilization AND little memory in use, sampled several times). Jobs wait in a
queue; nothing is ever started on a GPU someone else is using. Finished runs
(summary.json) are skipped and interrupted ones resume from latest.pt, so the
queue can simply be restarted. Every training process runs under nice/ionice.

    python -m lstm.queue pilot      --budget-tokens 2000000
    python -m lstm.queue tuning     --tuning-tokens 50000000
    python -m lstm.queue production --learning-rate 0.001 --dropout 0.1 --sizes s m m_deep l
    python -m lstm.queue tuning --tuning-tokens 50000000 --dry-run     # print the plan only

Monitor with: python -m comparison.status --roots lstm/runs --watch 60
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from comparison.protocol import load_protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "lstm" / "runs"


@dataclass
class Job:
    name: str
    run_dir: Path
    args: list[str]
    gpu: int | None = None
    process: subprocess.Popen | None = field(default=None, repr=False)


def plan(command: str, options: argparse.Namespace) -> list[Job]:
    protocol = load_protocol()
    tuning = protocol["tuning"]
    common = ["--architecture", "lstm"]
    jobs = []
    if command == "pilot":
        for size in options.sizes:
            run_dir = RUN_ROOT / "pilot" / size
            jobs.append(Job(f"pilot {size}", run_dir, [*common, "--mode", "exploratory", "--size", size,
                            "--budget-tokens", str(options.budget_tokens), "--learning-rate", "0.001",
                            "--dropout", "0.1", "--seed", str(protocol["training"]["tuning_seed"])]))
    elif command == "tuning":
        size = protocol["tuning"]["size"]
        for lr, dropout in itertools.product(tuning["learning_rates"], tuning["dropouts"]):
            run_dir = RUN_ROOT / "tuning" / f"{size}_lr{lr:g}_d{dropout:g}"
            jobs.append(Job(f"tune lr={lr:g} d={dropout:g}", run_dir, [
                *common, "--mode", "exploratory", "--size", size, "--budget-tokens", str(options.tuning_tokens),
                "--learning-rate", str(lr), "--dropout", str(dropout),
                "--seed", str(protocol["training"]["tuning_seed"])]))
    elif command == "production":
        for size, seed in itertools.product(options.sizes, protocol["training"]["seeds"]):
            run_dir = RUN_ROOT / "production" / size / f"seed{seed}"
            jobs.append(Job(f"prod {size} seed {seed}", run_dir, [
                *common, "--mode", "production", "--size", size, "--snapshot-file", str(options.snapshot_file),
                "--learning-rate", str(options.learning_rate), "--dropout", str(options.dropout),
                "--seed", str(seed)]))
    return jobs


def idle_gpus(max_util: int, max_memory_mib: int, samples: int = 3) -> set[int]:
    """GPUs idle in EVERY sample (avoids grabbing a GPU that is only momentarily quiet)."""
    idle = None
    for sample in range(samples):
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
        now = {int(i) for i, util, memory in (line.split(", ") for line in output.strip().splitlines())
               if int(util) <= max_util and int(memory) <= max_memory_mib}
        idle = now if idle is None else idle & now
        if sample + 1 < samples:
            time.sleep(2)
    return idle or set()


def log_path(job: Job) -> Path:
    return job.run_dir.with_name(job.run_dir.name + ".log")


def launch(job: Job, gpu: int, options: argparse.Namespace) -> None:
    # The trainer refuses a non-empty new run folder, so the log lives next to it.
    job.run_dir.parent.mkdir(parents=True, exist_ok=True)
    resume = ["--resume"] if (job.run_dir / "latest.pt").is_file() else []
    command = ["nice", "-n", "19", "ionice", "-c3", sys.executable, "-m", "transformer.train",
               *job.args, "--run-dir", str(job.run_dir), "--token-dir", str(options.token_dir), *resume]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4"}
    log = log_path(job).open("a", encoding="utf-8")
    log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} GPU {gpu}: {' '.join(command)}\n")
    log.flush()
    job.gpu = gpu
    job.process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
    print(f"[{time.strftime('%H:%M:%S')}] started {job.name} on GPU {gpu}" + (" (resume)" if resume else ""),
          flush=True)


def run_queue(jobs: list[Job], options: argparse.Namespace) -> int:
    pending = [job for job in jobs if not (job.run_dir / "summary.json").is_file()]
    print(f"{len(jobs) - len(pending)} already complete, {len(pending)} to run, max {options.max_gpus} GPUs",
          flush=True)
    running: list[Job] = []
    failures = 0
    while pending or running:
        for job in list(running):
            code = job.process.poll()
            if code is not None:
                running.remove(job)
                failures += code != 0
                print(f"[{time.strftime('%H:%M:%S')}] {'finished' if code == 0 else f'FAILED ({code})'} "
                      f"{job.name} (GPU {job.gpu}); log: {log_path(job)}", flush=True)
        if pending and len(running) < options.max_gpus:
            busy = {job.gpu for job in running}
            free = sorted(idle_gpus(options.max_util, options.max_memory_mib) - busy)
            for gpu in free[:options.max_gpus - len(running)]:
                job = pending.pop(0)
                launch(job, gpu, options)
                running.append(job)
                if not pending:
                    break
        time.sleep(options.poll_seconds)
    print(f"Queue finished: {failures} failed", flush=True)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["pilot", "tuning", "production"])
    parser.add_argument("--budget-tokens", type=int, default=2_000_000, help="pilot only")
    parser.add_argument("--tuning-tokens", type=int, help="tuning only: tokens per trial (same as the Transformer)")
    parser.add_argument("--sizes", nargs="+", default=None, help="pilot: s m l; production: s m m_deep l")
    parser.add_argument("--learning-rate", type=float, help="production only: tuning winner")
    parser.add_argument("--dropout", type=float, help="production only: tuning winner")
    parser.add_argument("--snapshot-file", type=Path, default=PROJECT_ROOT / "comparison/protocol.production.lock.json")
    parser.add_argument("--token-dir", type=Path, default=PROJECT_ROOT.parent / "dataset/processed/tokens")
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--max-util", type=int, default=5, help="max %% utilization to count a GPU as idle")
    parser.add_argument("--max-memory-mib", type=int, default=2048, help="max MiB in use to count a GPU as idle")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()
    if options.command == "tuning" and not options.tuning_tokens:
        parser.error("tuning needs --tuning-tokens (agree it with the Transformer side first)")
    if options.command == "production" and (options.learning_rate is None or options.dropout is None):
        parser.error("production needs --learning-rate and --dropout from the tuning winner")
    options.sizes = options.sizes or (["s", "m", "l"] if options.command == "pilot" else ["s", "m", "m_deep", "l"])
    jobs = plan(options.command, options)
    if options.dry_run:
        for job in jobs:
            print(job.name, "->", job.run_dir.relative_to(PROJECT_ROOT), " ".join(job.args))
        print("Idle GPUs now:", sorted(idle_gpus(options.max_util, options.max_memory_mib, samples=1)))
        return
    raise SystemExit(1 if run_queue(jobs, options) else 0)


if __name__ == "__main__":
    main()
