"""Bind the shared protocol to verified inputs; gate production on timing evidence."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

import yaml

from dataset.common import DEFAULT_DATA_ROOT, read_json, sha256_file, utc_now, write_json
from dataset.encode import load_cache
from dataset.windows import load_windows
from tokenizer.core import DEFAULT_OUTPUT, load_tokenizer

DEFAULT_PROTOCOL = Path(__file__).with_name("protocol.yaml")
DEFAULT_LOCK = Path(__file__).with_name("protocol.lock.json")


def positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_protocol(path: Path = DEFAULT_PROTOCOL) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        required = {"schema_version", "name", "stage", "inputs", "windows", "matching", "training",
                    "budget", "selection", "tuning", "evaluation", "hypotheses"}
        if not isinstance(config, dict) or set(config) != required or config["schema_version"] != 1:
            raise ValueError("Invalid protocol schema")
        if config["inputs"]["vocab_size"] != 16384:
            raise ValueError("The shared protocol requires the production 16,384-ID tokenizer")
        if any(config["windows"][key] != value for key, value in {
            "targets": 512, "stride": 512, "article_boundaries": "never_cross",
            "eos": "exactly_one_per_article", "bos": "restart_each_window", "padding_id": 0,
            "reset_transformer_positions": "each_window", "reset_lstm_hidden_and_cell": "each_window",
            "drop_last": False,
        }.items()):
            raise ValueError("Window protocol conflicts with the implemented shared pipeline")
        training, tuning = config["training"], config["tuning"]
        seeds = training["seeds"]
        if (len(seeds) != 3 or len(set(seeds)) != 3 or any(type(s) is not int or s < 0 for s in seeds)
                or type(training["tuning_seed"]) is not int or training["tuning_seed"] < 0
                or training["tuning_seed"] in seeds):
            raise ValueError("Require three distinct production seeds and a separate tuning seed")
        positive_int(training["effective_batch_windows"], "Effective batch")
        if training["precision"] != "float32":
            raise ValueError("This protocol freezes float32; revise and revalidate before changing precision")
        if (len(tuning["learning_rates"]) * len(tuning["dropouts"]) != 8
                or tuning["trials_per_architecture"] != 8
                or len(set(tuning["learning_rates"])) != len(tuning["learning_rates"])
                or len(set(tuning["dropouts"])) != len(tuning["dropouts"])
                or any(lr <= 0 for lr in tuning["learning_rates"])
                or any(not 0 <= d < 1 for d in tuning["dropouts"])):
            raise ValueError("Require eight distinct valid LR/dropout trials per architecture")
        budget = config["budget"]
        positive_int(budget["target_prediction_tokens"], "Target budget")
        if not budget["pilot_required_before_production"] or any(budget[k] is not None for k in (
            "exact_prediction_tokens", "tuning_prediction_tokens", "microbatch_windows", "gradient_accumulation_steps",
        )):
            raise ValueError("Exact run settings must be supplied by recorded pilot evidence")
        if config["selection"]["split"] != "validation" or config["inputs"]["test_use"] != "final_reporting_only":
            raise ValueError("Test data cannot be used for selection")
        positive_int(config["selection"]["validation_every_prediction_tokens"], "Validation interval")
        if (config["matching"]["non_embedding_relative_tolerance"] != 0.05
                or config["hypotheses"]["h2"]["equivalence_margin_percentage_points"] != 2.0):
            raise ValueError("Parameter matching/equivalence margins differ from the plan")
    except (KeyError, TypeError) as error:
        raise ValueError(f"Incomplete or invalid protocol: {error}") from error
    return config


def apply_pilot(config: dict, pilot: dict) -> dict:
    """Validate measured Stage 5 evidence and fill only the deferred budget fields."""
    if pilot.get("status") != "PASS" or not pilot.get("hardware"):
        raise ValueError("Pilot must record PASS and hardware identity")
    measurements = pilot.get("measurements", [])
    expected = {(architecture, size) for architecture in ("transformer", "lstm") for size in ("s", "m", "l")}
    observed = set()
    for row in measurements:
        pair = (row.get("architecture"), row.get("size"))
        if pair in observed or pair not in expected:
            raise ValueError("Pilot contains repeated or unexpected model measurements")
        observed.add(pair)
        for name in ("prediction_tokens_per_second", "peak_memory_bytes", "validation_seconds", "checkpoint_bytes"):
            value = row.get(name)
            if type(value) not in (float, int) or not 0 < value < float("inf"):
                raise ValueError(f"Missing/invalid pilot measurement: {name}")
    if observed != expected:
        raise ValueError("Pilot must measure both architectures at S/M/L")
    result = copy.deepcopy(config)
    settings = pilot.get("settings", {})
    for key in ("exact_prediction_tokens", "tuning_prediction_tokens", "microbatch_windows", "gradient_accumulation_steps"):
        result["budget"][key] = positive_int(settings.get(key), key)
    if (result["budget"]["microbatch_windows"] * result["budget"]["gradient_accumulation_steps"]
            != config["training"]["effective_batch_windows"]):
        raise ValueError("Microbatch times accumulation must equal the shared effective batch")
    result["stage"] = "production_ready_after_pilot"
    return result


def verify_pilot_inputs(pilot: dict, protocol: Path, token_root: Path, window_root: Path) -> None:
    for name, path in {
        "protocol_sha256": protocol, "token_manifest_sha256": token_root / "manifest.json",
        "window_manifest_sha256": window_root / "manifest.json",
    }.items():
        if pilot.get(name) != sha256_file(path):
            raise ValueError(f"Pilot evidence is not bound to these inputs: {name}")


def freeze(token_root: Path, window_root: Path | None = None, *, artifacts: Path = DEFAULT_OUTPUT,
           protocol: Path = DEFAULT_PROTOCOL, output: Path = DEFAULT_LOCK, pilot_path: Path | None = None) -> dict:
    output, token_root = Path(output).resolve(), Path(token_root).resolve()
    window_root = Path(window_root).resolve() if window_root else token_root / "windows"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite protocol lock: {output}; choose --output")
    config = load_protocol(protocol)
    load_tokenizer(artifacts)
    tokenizer = read_json(Path(artifacts) / "manifest.json")
    cache = load_cache(token_root)
    windows = load_windows(window_root, token_root)
    if not (config["inputs"]["tokenizer_sha256"] == tokenizer["tokenizer_sha256"]
            == cache["tokenizer_sha256"] == windows["tokenizer_sha256"]
            and config["inputs"]["vocab_size"] == cache["vocab_size"]):
        raise ValueError("Protocol/tokenizer/cache/window identities disagree")
    pilot = read_json(pilot_path) if pilot_path else None
    if pilot is not None:
        verify_pilot_inputs(pilot, protocol, token_root, window_root)
    configuration = apply_pilot(config, pilot) if pilot is not None else config
    result = {
        "schema_version": 1, "created_at": utc_now(), "production_ready": pilot is not None,
        "configuration": configuration, "protocol_sha256": sha256_file(protocol),
        "token_manifest_sha256": sha256_file(token_root / "manifest.json"),
        "window_manifest_sha256": sha256_file(window_root / "manifest.json"),
        "pilot": pilot, "pilot_sha256": sha256_file(pilot_path) if pilot_path else None,
        "code_sha256": {name: sha256_file(path) for name, path in {
            "encode.py": Path(__file__).parents[1] / "dataset/encode.py",
            "windows.py": Path(__file__).parents[1] / "dataset/windows.py",
            "protocol.py": Path(__file__),
        }.items()},
    }
    # Hard-link publication is atomic and fails if another process already created the lock.
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".protocol-", dir=output.parent) as temporary:
        staging = Path(temporary) / "lock.json"
        write_json(staging, result)
        os.link(staging, output)
    return result


def check_lock(path: Path, token_root: Path, window_root: Path | None = None, *,
               protocol: Path = DEFAULT_PROTOCOL, artifacts: Path = DEFAULT_OUTPUT,
               require_production: bool = False) -> dict:
    lock = read_json(path)
    token_root = Path(token_root)
    window_root = Path(window_root) if window_root else token_root / "windows"
    cache = load_cache(token_root)
    windows = load_windows(window_root, token_root)
    config = load_protocol(protocol)
    tokenizer = load_tokenizer(artifacts)
    if (config["inputs"]["tokenizer_sha256"] != sha256_file(Path(artifacts) / "tokenizer.json")
            or config["inputs"]["tokenizer_sha256"] != cache["tokenizer_sha256"]
            or config["inputs"]["tokenizer_sha256"] != windows["tokenizer_sha256"]
            or tokenizer.get_vocab_size() != config["inputs"]["vocab_size"]):
        raise ValueError("Protocol lock tokenizer/input identities disagree")
    expected_config = apply_pilot(config, lock["pilot"]) if lock.get("pilot") is not None else config
    if lock.get("pilot") is not None:
        verify_pilot_inputs(lock["pilot"], protocol, token_root, window_root)
    if (lock.get("schema_version") != 1 or lock.get("configuration") != expected_config
            or lock.get("production_ready") != (lock.get("pilot") is not None)
            or lock["protocol_sha256"] != sha256_file(protocol)
            or lock["token_manifest_sha256"] != sha256_file(token_root / "manifest.json")
            or lock["window_manifest_sha256"] != sha256_file(window_root / "manifest.json")):
        raise ValueError("Protocol lock is stale or inconsistent with current inputs")
    if set(lock.get("code_sha256", {})) != {"encode.py", "windows.py", "protocol.py"}:
        raise ValueError("Protocol lock has incomplete code fingerprints")
    for name, digest in lock["code_sha256"].items():
        directory = Path(__file__).parent if name == "protocol.py" else Path(__file__).parents[1] / "dataset"
        if name not in {"encode.py", "windows.py", "protocol.py"} or sha256_file(directory / name) != digest:
            raise ValueError("Shared pipeline code changed since protocol freeze")
    if require_production and not lock["production_ready"]:
        raise ValueError("Production is blocked until Stage 5 timing evidence freezes exact budgets and microbatches")
    return lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-production", action="store_true")
    args = parser.parse_args()
    if args.require_production and not args.check:
        parser.error("--require-production requires --check")
    if args.check and args.pilot:
        parser.error("--pilot is used when creating a new lock, not checking an existing one")
    result = (check_lock(args.output, args.tokens, args.windows, protocol=args.protocol,
                         artifacts=args.artifacts, require_production=args.require_production) if args.check else
              freeze(args.tokens, args.windows, artifacts=args.artifacts, protocol=args.protocol,
                     output=args.output, pilot_path=args.pilot))
    print(json.dumps({"production_ready": result["production_ready"], "lock": str(args.output)}))


if __name__ == "__main__":
    main()
