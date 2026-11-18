"""Validation-only size-S tuning with independent, resumable Slurm array trials.

Uses the unchanged exploratory trainer with an explicit equal tuning budget.
This study does not unlock production or use the held-out test/BLiMP scores.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import torch

from comparison.protocol import DEFAULT_PROTOCOL, check_lock, freeze, load_protocol, positive_int
from dataset.common import DEFAULT_DATA_ROOT, PROJECT_ROOT, read_json, sha256_file, utc_now, write_json
from tokenizer.core import DEFAULT_OUTPUT


def code_hashes() -> dict[str, str]:
    names = ("transformer/model.py", "transformer/train.py", "transformer/tuning.py",
             "transformer/tune_s.sbatch", "dataset/model_inputs.py", "tokenizer/core.py")
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def trial_grid(config: dict) -> list[dict]:
    return [{"index": index, "learning_rate": lr, "dropout": dropout,
             "directory": f"trial_{index:02d}"}
            for index, (lr, dropout) in enumerate(
                (lr, dropout) for lr in config["tuning"]["learning_rates"]
                for dropout in config["tuning"]["dropouts"])]


def verify_study(root: Path) -> dict:
    study = read_json(root / "study.json")
    if study["code_sha256"] != code_hashes():
        raise ValueError("Tuning code changed; retain this study and prepare a new study directory")
    if study["torch_version"] != torch.__version__ or study["cuda_version"] != torch.version.cuda:
        raise ValueError("Tuning runtime changed; use the original environment or a new study")
    if sha256_file(root / "protocol.lock.json") != study["snapshot_sha256"]:
        raise ValueError("Tuning snapshot changed")
    lock = check_lock(root / "protocol.lock.json", Path(study["token_dir"]),
                      artifacts=Path(study["tokenizer_dir"]), protocol=DEFAULT_PROTOCOL)
    if study["trials"] != trial_grid(lock["configuration"]):
        raise ValueError("Tuning grid differs from the saved protocol")
    return study


def prepare(root: Path, *, budget_tokens: int, microbatch_windows: int,
            token_dir: Path, tokenizer_dir: Path) -> dict:
    budget_tokens = positive_int(budget_tokens, "Tuning budget")
    microbatch_windows = positive_int(microbatch_windows, "Microbatch")
    config = load_protocol()
    effective = config["training"]["effective_batch_windows"]
    if effective % microbatch_windows:
        raise ValueError("Microbatch must divide the effective batch")
    settings = {
        "schema_version": 1, "purpose": "transformer_s_tuning", "size": "s",
        "budget_tokens": budget_tokens, "microbatch_windows": microbatch_windows,
        "effective_batch_windows": effective, "seed": config["training"]["tuning_seed"],
        "token_dir": str(token_dir.resolve()), "tokenizer_dir": str(tokenizer_dir.resolve()),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "code_sha256": code_hashes(), "trials": trial_grid(config),
    }
    root = root.resolve()
    if root.exists():
        study = verify_study(root)
        if any(study.get(key) != value for key, value in settings.items()):
            raise ValueError("Existing study has different settings; choose another directory")
        return study
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".tuning-", dir=root.parent) as temporary:
        staging = Path(temporary) / "study"
        staging.mkdir()
        freeze(token_dir, artifacts=tokenizer_dir, protocol=DEFAULT_PROTOCOL,
               output=staging / "protocol.lock.json")
        # Keep readable rules alongside the fingerprinted preparation snapshot.
        (staging / "protocol.yaml").write_bytes(DEFAULT_PROTOCOL.read_bytes())
        study = {**settings, "created_at": utc_now(),
                 "snapshot_sha256": sha256_file(staging / "protocol.lock.json")}
        write_json(staging / "study.json", study)
        staging.rename(root)
    return study


def trainer_command(root: Path, study: dict, trial: dict, *, device: str) -> list[str]:
    command = [sys.executable, "-u", "-m", "transformer.train", "--mode", "exploratory",
               "--size", "s", "--seed", str(study["seed"]),
               "--budget-tokens", str(study["budget_tokens"]),
               "--microbatch-windows", str(study["microbatch_windows"]),
               "--learning-rate", str(trial["learning_rate"]), "--dropout", str(trial["dropout"]),
               "--token-dir", study["token_dir"], "--tokenizer-dir", study["tokenizer_dir"],
               "--run-dir", str(root / trial["directory"]), "--save-every-updates", "25",
               "--device", device]
    if (root / trial["directory"] / "latest.pt").is_file():
        command.append("--resume")
    return command


def completed_result(root: Path, study: dict, trial: dict) -> dict | None:
    directory = root / trial["directory"]
    if not (directory / "summary.json").is_file():
        return None
    result = read_json(directory / "summary.json")
    identity = read_json(directory / "run.json")["identity"]
    lock = read_json(root / "protocol.lock.json")
    expected = {"mode": "exploratory", "size": "s", "seed": study["seed"],
                "budget_tokens": study["budget_tokens"], "microbatch_windows": study["microbatch_windows"],
                "effective_batch_windows": study["effective_batch_windows"],
                "learning_rate": trial["learning_rate"], "torch_version": study["torch_version"],
                "cuda_version": study["cuda_version"], "protocol_sha256": lock["protocol_sha256"],
                "tokenizer_sha256": lock["configuration"]["inputs"]["tokenizer_sha256"],
                "token_manifest_sha256": lock["token_manifest_sha256"],
                "window_manifest_sha256": lock["window_manifest_sha256"],
                "model_code_sha256": study["code_sha256"]["transformer/model.py"],
                "train_code_sha256": study["code_sha256"]["transformer/train.py"]}
    if (any(identity.get(key) != value for key, value in expected.items())
            or identity["model"]["dropout"] != trial["dropout"]):
        raise ValueError(f"Trial provenance differs: {directory}")
    if (result["status"] != "complete" or result["consumed_tokens"] != study["budget_tokens"]
            or not math.isfinite(result["best_validation_nll"])
            or not 0 <= result["best_checkpoint_tokens"] <= study["budget_tokens"]
            or not all((directory / name).is_file() for name in ("best.pt", "final.pt", "latest.pt"))):
        raise ValueError(f"Invalid completed trial: {directory}")
    return result


def run_trial(root: Path, index: int, *, device: str = "cuda") -> int:
    root = root.resolve()
    study = verify_study(root)
    if not 0 <= index < len(study["trials"]):
        raise ValueError("Trial index outside the frozen grid")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Tuning requires a visible allocated CUDA GPU")
    trial = study["trials"][index]
    # Lock outside the run directory: the trainer refuses nonempty new run directories.
    with (root / f"trial_{index:02d}.lock").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This trial already has a writer") from error
        if completed_result(root, study, trial) is not None:
            print(f"Trial {index} is already complete; keeping its artifacts", flush=True)
            return 0
        command = trainer_command(root, study, trial, device=device)
        attempt_dir = root / "attempts"
        attempt_dir.mkdir(exist_ok=True)
        attempt = attempt_dir / f"trial_{index:02d}_{time.time_ns()}.json"
        record = {"status": "running", "started_at": utc_now(), "trial_index": index,
                  "command": command, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                  "device": device}
        write_json(attempt, record)
        print(json.dumps(record), flush=True)
        # Inherit the Slurm task's device mapping, stdout and stderr without alteration.
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            valid = completed_result(root, study, trial) if result.returncode == 0 else None
            code = result.returncode if result.returncode else (0 if valid is not None else 1)
        except Exception as error:
            write_json(attempt, {**record, "status": "failed", "error": str(error),
                                "finished_at": utc_now()})
            raise
        write_json(attempt, {**record, "status": "complete" if code == 0 else "failed",
                             "returncode": code, "finished_at": utc_now()})
        return code


def collect(root: Path) -> dict:
    study = verify_study(root)
    rows = []
    for trial in study["trials"]:
        result = completed_result(root, study, trial)
        attempts = sorted((root / "attempts").glob(f"trial_{trial['index']:02d}_*.json"))
        last = read_json(attempts[-1]) if attempts else {}
        status = "complete" if result else last.get("status", "pending")
        rows.append({**trial, "status": status, "result": result,
                     "last_attempt": str(attempts[-1]) if attempts else None})
    resolved = all(row["status"] in ("complete", "failed") for row in rows)
    completed = [row for row in rows if row["status"] == "complete"]
    winner = min(completed, key=lambda row: (row["result"]["best_validation_nll"], row["index"])) \
        if resolved and completed else None
    return {"status": "finished" if resolved else "incomplete",
            "selection": "lowest_validation_nll_then_grid_order", "trials": rows,
            "winner": winner, "failed_trials": sum(row["status"] == "failed" for row in rows),
            "caveat": "Failed trials are retained; investigate infrastructure failures before accepting selection.",
            "study_sha256": sha256_file(root / "study.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    preparation = subparsers.add_parser("prepare", help="Freeze inputs, grid and explicit tuning budget")
    preparation.add_argument("--study-dir", type=Path, required=True)
    preparation.add_argument("--budget-tokens", type=int, required=True)
    preparation.add_argument("--microbatch-windows", type=int, default=16)
    preparation.add_argument("--token-dir", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    preparation.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_OUTPUT)
    running = subparsers.add_parser("run", help="Run or resume one grid trial")
    running.add_argument("--study-dir", type=Path, required=True)
    running.add_argument("--trial-index", type=int, required=True)
    running.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    collection = subparsers.add_parser("collect", help="Print all results and the validation-selected winner")
    collection.add_argument("--study-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.study_dir, budget_tokens=args.budget_tokens,
                         microbatch_windows=args.microbatch_windows, token_dir=args.token_dir,
                         tokenizer_dir=args.tokenizer_dir)
    elif args.action == "run":
        raise SystemExit(run_trial(args.study_dir, args.trial_index, device=args.device))
    else:
        result = collect(args.study_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
