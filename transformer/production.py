"""Freeze and run the proposal's nine Transformer experiments with two Slurm GPUs.

The existing trainer remains authoritative for optimization and resume identity.
Joint production requires the shared timing gate. An explicitly requested
Transformer-only campaign freezes its own budget without claiming joint readiness.
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

from comparison.protocol import DEFAULT_PROTOCOL, check_lock, load_protocol, positive_int
from dataset.common import PROJECT_ROOT, read_json, sha256_file, utc_now, write_json
from transformer import tuning


DEFAULT_TUNING = PROJECT_ROOT / "transformer/runs/s/tuning_50M_seed101_6lr"
DEFAULT_CAMPAIGN = PROJECT_ROOT / "transformer/runs/production_3seed"


def code_hashes() -> dict[str, str]:
    names = ("transformer/production.py", "transformer/production.sbatch", "transformer/run_production.sh",
             "transformer/train.py", "transformer/model.py", "dataset/model_inputs.py",
             "tokenizer/core.py")
    return {name: sha256_file(PROJECT_ROOT / name) for name in names}


def run_grid(config: dict) -> list[dict]:
    # Launch long L runs first to reduce the tail with two independent workers.
    return [{"index": index, "size": size, "seed": seed,
             "directory": f"{size}/seed{seed}"}
            for index, (size, seed) in enumerate(
                (size, seed) for size in ("l", "m", "s") for seed in config["training"]["seeds"])]


def selected_tuning(root: Path) -> tuple[dict, dict]:
    selection = tuning.collect(root)
    if (selection["status"] != "finished" or selection["failed_trials"]
            or selection["winner"] is None
            or any(row["status"] != "complete" for row in selection["trials"])):
        raise ValueError("Require every tuning trial to complete successfully before production")
    return selection, tuning.verify_study(root)


def plan(tuning_root: Path) -> dict:
    selection, study = selected_tuning(tuning_root.resolve())
    config = load_protocol()
    return {"status": "planned_awaiting_joint_pilot", "production_ready": False,
            "learning_rate": selection["winner"]["learning_rate"],
            "dropout": selection["winner"]["dropout"], "precision": config["training"]["precision"],
            "proposed_prediction_tokens_per_run": config["budget"]["target_prediction_tokens"],
            "tuning_prediction_tokens_per_trial": study["budget_tokens"],
            "effective_batch_windows": config["training"]["effective_batch_windows"],
            "runs": run_grid(config), "concurrent_runs": 2, "gpus_per_run": 1,
            "next_step": "Freeze a new shared production snapshot with the joint pilot; then prepare."}


def prepare(root: Path, *, tuning_root: Path, snapshot: Path) -> dict:
    root, tuning_root, snapshot = root.resolve(), tuning_root.resolve(), snapshot.resolve()
    selection, study = selected_tuning(tuning_root)
    lock = check_lock(snapshot, Path(study["token_dir"]), artifacts=Path(study["tokenizer_dir"]),
                      require_production=True)
    config = lock["configuration"]
    tuning_lock = read_json(tuning_root / "protocol.lock.json")
    if any(lock[key] != tuning_lock[key] for key in (
            "protocol_sha256", "token_manifest_sha256", "window_manifest_sha256")):
        raise ValueError("Production and tuning inputs/protocol differ")
    if config["budget"]["tuning_prediction_tokens"] != study["budget_tokens"]:
        raise ValueError("Joint pilot tuning budget differs from the completed Transformer study")
    settings = {"schema_version": 1, "purpose": "transformer_production_three_seeds",
                "budget_tokens": config["budget"]["exact_prediction_tokens"],
                "microbatch_windows": config["budget"]["microbatch_windows"],
                "gradient_accumulation_steps": config["budget"]["gradient_accumulation_steps"],
                "effective_batch_windows": config["training"]["effective_batch_windows"],
                "precision": config["training"]["precision"],
                "learning_rate": selection["winner"]["learning_rate"],
                "dropout": selection["winner"]["dropout"],
                "token_dir": study["token_dir"], "tokenizer_dir": study["tokenizer_dir"],
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "code_sha256": code_hashes(), "snapshot_sha256": sha256_file(snapshot),
                "tuning_study_sha256": sha256_file(tuning_root / "study.json"),
                "runs": run_grid(config), "save_every_updates": 100}
    if root.exists():
        previous = verify(root)
        if any(previous.get(key) != value for key, value in settings.items()):
            raise ValueError("Existing campaign settings differ; choose a new directory")
        return previous
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".production-", dir=root.parent) as temporary:
        staging = Path(temporary) / "campaign"
        staging.mkdir()
        (staging / "protocol.lock.json").write_bytes(snapshot.read_bytes())
        (staging / "protocol.yaml").write_bytes(DEFAULT_PROTOCOL.read_bytes())
        write_json(staging / "tuning_selection.json", selection)
        settings.update(created_at=utc_now(),
                        selection_sha256=sha256_file(staging / "tuning_selection.json"))
        write_json(staging / "campaign.json", settings)
        staging.rename(root)
    return settings


def prepare_standalone(root: Path, *, tuning_root: Path, budget_tokens: int,
                       microbatch_windows: int) -> dict:
    """Record the user's explicit decision to run Transformer before the joint pilot.

    Use the unchanged exploratory trainer with explicit frozen execution settings;
    do not fabricate pilot evidence or alter the shared production readiness flag.
    """
    root, tuning_root = root.resolve(), tuning_root.resolve()
    budget_tokens = positive_int(budget_tokens, "Transformer-only budget")
    microbatch_windows = positive_int(microbatch_windows, "Microbatch")
    selection, study = selected_tuning(tuning_root)
    snapshot = tuning_root / "protocol.lock.json"
    lock = check_lock(snapshot, Path(study["token_dir"]), artifacts=Path(study["tokenizer_dir"]))
    config = lock["configuration"]
    effective = config["training"]["effective_batch_windows"]
    if effective % microbatch_windows:
        raise ValueError("Microbatch must divide the frozen effective batch")
    execution = {"trainer_mode": "exploratory", "purpose": "transformer_only_three_seeds",
                 "joint_production_ready": False, "budget_tokens": budget_tokens,
                 "microbatch_windows": microbatch_windows,
                 "gradient_accumulation_steps": effective // microbatch_windows,
                 "pilot_requirement": "user_requested_transformer_only_before_joint_pilot"}
    settings = {**execution, "schema_version": 1, "effective_batch_windows": effective,
                "precision": config["training"]["precision"],
                "learning_rate": selection["winner"]["learning_rate"],
                "dropout": selection["winner"]["dropout"],
                "token_dir": study["token_dir"], "tokenizer_dir": study["tokenizer_dir"],
                "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                "code_sha256": code_hashes(), "snapshot_sha256": sha256_file(snapshot),
                "tuning_study_sha256": sha256_file(tuning_root / "study.json"),
                "runs": run_grid(config), "save_every_updates": 100}
    if root.exists():
        previous = verify(root)
        if any(previous.get(key) != value for key, value in settings.items()):
            raise ValueError("Existing campaign settings differ; choose a new directory")
        return previous
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".transformer-only-", dir=root.parent) as temporary:
        staging = Path(temporary) / "campaign"
        staging.mkdir()
        (staging / "protocol.lock.json").write_bytes(snapshot.read_bytes())
        (staging / "protocol.yaml").write_bytes(DEFAULT_PROTOCOL.read_bytes())
        write_json(staging / "execution.json", execution)
        write_json(staging / "tuning_selection.json", selection)
        settings.update(created_at=utc_now(), execution_sha256=sha256_file(staging / "execution.json"),
                        selection_sha256=sha256_file(staging / "tuning_selection.json"))
        write_json(staging / "campaign.json", settings)
        staging.rename(root)
    return settings


def verify(root: Path) -> dict:
    settings = read_json(root / "campaign.json")
    if (settings["code_sha256"] != code_hashes()
            or settings["torch_version"] != torch.__version__
            or settings["cuda_version"] != torch.version.cuda):
        raise ValueError("Production code/runtime changed; use the frozen environment/checkout")
    if (sha256_file(root / "protocol.lock.json") != settings["snapshot_sha256"]
            or sha256_file(root / "tuning_selection.json") != settings["selection_sha256"]):
        raise ValueError("Campaign snapshot or tuning selection changed")
    standalone = settings.get("trainer_mode") == "exploratory"
    lock = check_lock(root / "protocol.lock.json", Path(settings["token_dir"]),
                      artifacts=Path(settings["tokenizer_dir"]), require_production=not standalone)
    config = lock["configuration"]
    winner = read_json(root / "tuning_selection.json")["winner"]
    expected = {"budget_tokens": config["budget"]["exact_prediction_tokens"],
                "microbatch_windows": config["budget"]["microbatch_windows"],
                "gradient_accumulation_steps": config["budget"]["gradient_accumulation_steps"],
                "effective_batch_windows": config["training"]["effective_batch_windows"],
                "precision": config["training"]["precision"], "runs": run_grid(config),
                "learning_rate": winner["learning_rate"], "dropout": winner["dropout"]}
    if standalone:
        if sha256_file(root / "execution.json") != settings["execution_sha256"]:
            raise ValueError("Transformer-only execution settings changed")
        execution = read_json(root / "execution.json")
        positive_int(execution["budget_tokens"], "Transformer-only budget")
        batch = positive_int(execution["microbatch_windows"], "Microbatch")
        if (execution.get("purpose") != "transformer_only_three_seeds"
                or execution.get("joint_production_ready") is not False
                or execution.get("trainer_mode") != "exploratory"
                or execution.get("pilot_requirement") != "user_requested_transformer_only_before_joint_pilot"
                or batch * execution["gradient_accumulation_steps"] != expected["effective_batch_windows"]):
            raise ValueError("Invalid Transformer-only execution contract")
        expected.update(execution)
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError("Campaign settings differ from the frozen protocol/tuning winner")
    return settings


def expected_identity(root: Path, settings: dict, task: dict) -> dict:
    lock = read_json(root / "protocol.lock.json")
    expected = {"mode": settings.get("trainer_mode", "production"), "size": task["size"], "seed": task["seed"],
            **{key: settings[key] for key in ("budget_tokens", "microbatch_windows",
                                             "effective_batch_windows", "learning_rate",
                                             "torch_version", "cuda_version", "snapshot_sha256")},
            **{key: lock[key] for key in ("protocol_sha256", "token_manifest_sha256",
                                         "window_manifest_sha256")},
            "tokenizer_sha256": lock["configuration"]["inputs"]["tokenizer_sha256"],
            "model_code_sha256": settings["code_sha256"]["transformer/model.py"],
            "train_code_sha256": settings["code_sha256"]["transformer/train.py"]}
    if settings.get("trainer_mode") == "exploratory":
        expected["snapshot_sha256"] = None  # Existing trainer records no production lock in this mode.
    return expected


def completed(root: Path, settings: dict, task: dict) -> dict | None:
    directory = root / task["directory"]
    if not (directory / "summary.json").is_file():
        return None
    summary = read_json(directory / "summary.json")
    identity = read_json(directory / "run.json")["identity"]
    if (any(identity.get(key) != value for key, value in expected_identity(root, settings, task).items())
            or identity["model"]["dropout"] != settings["dropout"]):
        raise ValueError(f"Completed run provenance differs: {directory}")
    if (summary["status"] != "complete" or summary["mode"] != settings.get("trainer_mode", "production")
            or summary["size"] != task["size"] or summary["consumed_tokens"] != settings["budget_tokens"]
            or not math.isfinite(summary["best_validation_nll"])
            or not 0 <= summary["best_checkpoint_tokens"] <= settings["budget_tokens"]
            or not all((directory / name).is_file() and (directory / name).stat().st_size > 0
                       for name in ("best.pt", "final.pt", "latest.pt"))):
        raise ValueError(f"Invalid completed production run: {directory}")
    return summary


def trainer_command(root: Path, settings: dict, task: dict, device: str) -> list[str]:
    directory = root / task["directory"]
    command = [sys.executable, "-u", "-m", "transformer.train", "--mode", settings.get("trainer_mode", "production"),
               "--size", task["size"], "--seed", str(task["seed"]),
               "--snapshot-file", str(root / "protocol.lock.json"),
               "--learning-rate", str(settings["learning_rate"]), "--dropout", str(settings["dropout"]),
               "--token-dir", settings["token_dir"], "--tokenizer-dir", settings["tokenizer_dir"],
               "--run-dir", str(directory), "--save-every-updates", str(settings["save_every_updates"]),
               "--device", device]
    if settings.get("trainer_mode") == "exploratory":
        command.extend(["--budget-tokens", str(settings["budget_tokens"]),
                        "--microbatch-windows", str(settings["microbatch_windows"])])
    if (directory / "latest.pt").is_file():
        command.append("--resume")
    return command


def run_task(root: Path, index: int, *, device: str = "cuda") -> int:
    settings = verify(root)
    if not 0 <= index < len(settings["runs"]):
        raise ValueError("Run index is outside the frozen campaign")
    task = settings["runs"][index]
    with (root / f"run_{index:02d}.lock").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This production run already has a writer") from error
        if completed(root, settings, task) is not None:
            print(f"Already complete: {task['directory']}; keeping artifacts", flush=True)
            return 0
        if device == "cuda" and (not os.environ.get("SLURM_JOB_ID")
                                 or not os.environ.get("CUDA_VISIBLE_DEVICES")
                                 or not torch.cuda.is_available() or torch.cuda.device_count() != 1):
            raise RuntimeError("Production requires exactly one Slurm-assigned visible GPU per task")
        command = trainer_command(root, settings, task, device)
        attempt = root / "attempts" / f"run_{index:02d}_{time.time_ns()}.json"
        record = {"status": "running", "started_at": utc_now(), "task": task, "command": command,
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        write_json(attempt, record)
        print(json.dumps(record), flush=True)
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            valid = completed(root, settings, task) if result.returncode == 0 else None
            code = result.returncode or (0 if valid is not None else 1)
        except Exception as error:
            write_json(attempt, {**record, "status": "failed", "error": str(error), "finished_at": utc_now()})
            raise
        write_json(attempt, {**record, "status": "complete" if code == 0 else "failed",
                             "returncode": code, "finished_at": utc_now()})
        return code


def collect(root: Path) -> dict:
    settings = verify(root)
    rows = []
    for task in settings["runs"]:
        result = completed(root, settings, task)
        attempts = sorted((root / "attempts").glob(f"run_{task['index']:02d}_*.json"))
        last = read_json(attempts[-1]) if attempts else {}
        status = "complete" if result else last.get("status", "pending")
        rows.append({**task, "status": status, "result": result})
    return {"status": "complete" if all(row["status"] == "complete" for row in rows) else "incomplete",
            "completed_runs": sum(row["status"] == "complete" for row in rows), "expected_runs": len(rows),
            "runs": rows, "note": "A running attempt may have been interrupted; check Slurm. "
                                    "Test/context/copy/BLiMP evaluation is a separate step."}


def export_pilot(benchmark_root: Path, tuning_root: Path = DEFAULT_TUNING) -> dict:
    """Export only measured Transformer evidence for Hugo to combine with LSTM."""
    from transformer import batch_benchmark

    report = batch_benchmark.collect(benchmark_root)
    _, study = selected_tuning(tuning_root.resolve())
    if (read_json(benchmark_root / "benchmark.json")["tuning_study_sha256"]
            != sha256_file(tuning_root / "study.json")):
        raise ValueError("Benchmark and tuning study differ")
    recommendation = report["recommendation"]
    if report["status"] != "finished" or not recommendation:
        raise ValueError("Finish the batch benchmark before exporting Transformer evidence")
    batch = recommendation["microbatch_windows"]
    paths = [Path(row["report_path"]) for row in report["rows"]
             if row["microbatch_windows"] == batch]
    evidence = [read_json(path) for path in paths]
    first = evidence[0]
    return {"schema_version": 1, "status": "TRANSFORMER_ONLY", "production_ready": False,
            "created_at": utc_now(), "settings": first["settings"],
            **{key: first[key] for key in ("protocol_sha256", "token_manifest_sha256",
                                           "window_manifest_sha256", "tokenizer_sha256",
                                           "train_code_sha256", "model_code_sha256",
                                           "torch_version", "cuda_version", "timing_method")},
            "pilot_settings": {**first["pilot_settings"], "sizes": ["s", "m", "l"]},
            "hardware": [{"size": item["measurements"][0]["size"], **item["hardware"]}
                         for item in evidence],
            "measurements": [item["measurements"][0] for item in evidence],
            "source_reports": [{"path": str(path), "sha256": sha256_file(path)} for path in paths],
            "proposed_budgets": {"exact_prediction_tokens": load_protocol()["budget"]["target_prediction_tokens"],
                                 "tuning_prediction_tokens": study["budget_tokens"]},
            "limitations": "Transformer-only timing evidence at the fastest observed common microbatch. "
                           "Hugo must add LSTM S/M/L evidence and agree exact production/tuning budgets "
                           "and the shared microbatch before creating a PASS joint report."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("plan", "prepare", "prepare-standalone", "run", "collect", "check", "export-pilot"):
        command = sub.add_parser(name)
        if name in ("plan", "prepare", "prepare-standalone"):
            command.add_argument("--tuning-study", type=Path, default=DEFAULT_TUNING)
        if name not in ("plan", "export-pilot"):
            command.add_argument("--campaign-dir", type=Path, default=DEFAULT_CAMPAIGN)
        if name == "export-pilot":
            command.add_argument("--benchmark-dir", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--snapshot-file", type=Path, required=True)
        if name == "prepare-standalone":
            command.add_argument("--budget-tokens", type=int, default=250_000_000)
            command.add_argument("--microbatch-windows", type=int, default=64)
        if name == "run":
            command.add_argument("--run-index", type=int, required=True)
    args = parser.parse_args()
    if args.action == "plan":
        result = plan(args.tuning_study)
    elif args.action == "export-pilot":
        result = export_pilot(args.benchmark_dir.resolve())
    elif args.action == "prepare":
        result = prepare(args.campaign_dir, tuning_root=args.tuning_study, snapshot=args.snapshot_file)
    elif args.action == "prepare-standalone":
        result = prepare_standalone(args.campaign_dir, tuning_root=args.tuning_study,
                                    budget_tokens=args.budget_tokens, microbatch_windows=args.microbatch_windows)
    elif args.action == "run":
        raise SystemExit(run_task(args.campaign_dir.resolve(), args.run_index))
    else:
        result = collect(args.campaign_dir.resolve()) if args.action == "collect" else verify(args.campaign_dir.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
