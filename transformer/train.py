"""Train a shared-window Transformer; exploratory runs never count as production."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from comparison.protocol import DEFAULT_LOCK, DEFAULT_PROTOCOL, check_lock, load_protocol
from dataset.common import DEFAULT_DATA_ROOT, read_json, sha256_file, utc_now, write_json
from dataset.model_inputs import (
    NextTokenWindowDataset, compute_masked_cross_entropy, create_batch_loader,
    limit_loss_mask_to_budget,
)
from tokenizer.core import DEFAULT_OUTPUT as DEFAULT_TOKENIZER_DIR, load_tokenizer

from .model import TransformerConfig, TransformerLM


def set_seed(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def rng_state() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_torch(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_event(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps({"at": utc_now(), **row}, sort_keys=True) + "\n")
        output.flush()


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()
            if isinstance(value, torch.Tensor)}


def make_optimizer(model: TransformerLM, config: dict) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    settings = config["training"]["optimizer"]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=0.0, betas=tuple(settings["betas"]), eps=settings["eps"],
    )


def learning_rate(consumed: int, budget: int, peak: float, warmup_fraction: float,
                  minimum_ratio: float) -> float:
    warmup = max(1, round(budget * warmup_fraction))
    if consumed <= warmup:
        return peak * consumed / warmup
    progress = min(1.0, (consumed - warmup) / max(1, budget - warmup))
    return peak * (minimum_ratio + (1 - minimum_ratio) * (1 + math.cos(math.pi * progress)) / 2)


@torch.inference_mode()
def evaluate(model: TransformerLM, dataset: NextTokenWindowDataset, *, batch_size: int,
             device: torch.device) -> dict:
    model.eval()
    started = time.perf_counter()
    total_loss, total_count = 0.0, 0
    for raw in create_batch_loader(dataset, batch_size=batch_size, training=False):
        batch = move_batch(raw, device)
        logits = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"])
        loss, count = compute_masked_cross_entropy(logits, batch["target_ids"], batch["loss_mask"])
        total_loss += float(loss)
        total_count += int(count)
    if total_count == 0:
        raise ValueError("Validation contains no prediction targets")
    nll = total_loss / total_count
    return {"nll": nll, "perplexity": math.exp(nll), "tokens": total_count,
            "seconds": time.perf_counter() - started}


def checkpoint(model: TransformerLM, optimizer: torch.optim.Optimizer, *, identity: dict,
               epoch: int, cursor: int, consumed: int, updates: int, next_validation: int,
               best_nll: float, best_tokens: int, train_seconds: float) -> dict:
    return {
        "identity": identity, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": rng_state(), "epoch": epoch, "cursor": cursor, "consumed": consumed,
        "updates": updates, "next_validation": next_validation, "best_nll": best_nll,
        "best_tokens": best_tokens, "train_seconds": train_seconds,
    }


def run(args: argparse.Namespace) -> dict:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install a CUDA-enabled PyTorch build on the server")
    device = torch.device(args.device)
    if args.mode == "production":
        if args.budget_tokens is not None:
            raise ValueError("Production token budget comes only from the measured protocol snapshot")
        lock = check_lock(args.snapshot_file, args.token_dir, protocol=DEFAULT_PROTOCOL,
                          artifacts=args.tokenizer_dir, require_production=True)
        protocol = lock["configuration"]
        budget = protocol["budget"]["exact_prediction_tokens"]
        microbatch = protocol["budget"]["microbatch_windows"]
        if args.microbatch_windows is not None and args.microbatch_windows != microbatch:
            raise ValueError("Microbatch differs from the production snapshot")
        if args.seed not in protocol["training"]["seeds"]:
            raise ValueError("Production seed must be one of the frozen run seeds")
        if (args.learning_rate not in protocol["tuning"]["learning_rates"]
                or args.dropout not in protocol["tuning"]["dropouts"]):
            raise ValueError("Production hyperparameters must come from the frozen tuning grid")
        snapshot_hash = sha256_file(args.snapshot_file)
    else:
        protocol = load_protocol()
        if args.budget_tokens is None or args.budget_tokens <= 0:
            raise ValueError("Exploratory runs require --budget-tokens")
        budget = args.budget_tokens
        microbatch = args.microbatch_windows or 2
        snapshot_hash = None
        load_tokenizer(args.tokenizer_dir)
        if sha256_file(args.tokenizer_dir / "tokenizer.json") != protocol["inputs"]["tokenizer_sha256"]:
            raise ValueError("Tokenizer bundle differs from the frozen protocol")
    effective_batch = protocol["training"]["effective_batch_windows"]
    if microbatch <= 0 or effective_batch % microbatch:
        raise ValueError("Microbatch must divide the frozen effective batch of 64 windows")
    if not 0 <= args.dropout < 1 or args.learning_rate <= 0 or args.seed < 0:
        raise ValueError("Invalid dropout, learning rate, or seed")
    if args.save_every_updates <= 0:
        raise ValueError("--save-every-updates must be positive")
    config = TransformerConfig.for_size(args.size, dropout=args.dropout)
    if (config.vocab_size != protocol["inputs"]["vocab_size"]
            or config.width != protocol["matching"]["embedding_widths"][args.size]
            or config.max_positions != protocol["windows"]["targets"]):
        raise ValueError("Transformer architecture conflicts with the shared protocol")
    training_data = NextTokenWindowDataset(args.token_dir, "train")
    validation_data = NextTokenWindowDataset(args.token_dir, "validation")
    if training_data.manifest["tokenizer_sha256"] != protocol["inputs"]["tokenizer_sha256"]:
        raise ValueError("Saved windows do not use the frozen protocol tokenizer")
    identity = {
        "mode": args.mode, "size": args.size, "model": config.to_dict(),
        "seed": args.seed, "budget_tokens": budget, "microbatch_windows": microbatch,
        "effective_batch_windows": effective_batch, "learning_rate": args.learning_rate,
        "protocol_sha256": sha256_file(DEFAULT_PROTOCOL), "snapshot_sha256": snapshot_hash,
        "token_manifest_sha256": sha256_file(args.token_dir / "manifest.json"),
        "window_manifest_sha256": sha256_file(args.token_dir / "windows/manifest.json"),
        "tokenizer_sha256": sha256_file(args.tokenizer_dir / "tokenizer.json"),
        "model_code_sha256": sha256_file(Path(__file__).with_name("model.py")),
        "train_code_sha256": sha256_file(Path(__file__)),
    }
    run_dir = args.run_dir.resolve()
    if args.resume:
        if not (run_dir / "latest.pt").is_file():
            raise FileNotFoundError("--resume requires an existing latest.pt")
    else:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "run.json", {"created_at": utc_now(), "identity": identity})
    set_seed(args.seed)
    model = TransformerLM(config).to(device)
    optimizer = make_optimizer(model, protocol)
    if args.resume:
        saved = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        if saved.get("identity") != identity:
            raise ValueError("Checkpoint settings or input/code hashes differ from this run")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        epoch, cursor, consumed, updates = (saved[key] for key in ("epoch", "cursor", "consumed", "updates"))
        next_validation = saved["next_validation"]
        best_nll, best_tokens = saved["best_nll"], saved["best_tokens"]
        train_seconds = saved["train_seconds"]
    else:
        epoch = cursor = consumed = updates = 0
        next_validation = protocol["selection"]["validation_every_prediction_tokens"]
        best_nll, best_tokens = math.inf, 0
        train_seconds = 0.0
    events = run_dir / "metrics.jsonl"
    last_validation_seconds = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def save_latest() -> None:
        save_torch(run_dir / "latest.pt", checkpoint(
            model, optimizer, identity=identity, epoch=epoch, cursor=cursor,
            consumed=consumed, updates=updates, next_validation=next_validation,
            best_nll=best_nll, best_tokens=best_tokens, train_seconds=train_seconds,
        ))

    def validate() -> None:
        nonlocal best_nll, best_tokens, last_validation_seconds
        result = evaluate(model, validation_data, batch_size=microbatch, device=device)
        last_validation_seconds = result["seconds"]
        append_event(events, {"event": "validation", "consumed_tokens": consumed,
                              "updates": updates, **result})
        if result["nll"] < best_nll:
            best_nll, best_tokens = result["nll"], consumed
            save_torch(run_dir / "best.pt", {"identity": identity, "model": model.state_dict(),
                                                   "consumed_tokens": consumed, "validation": result})
        print(f"validation tokens={consumed} nll={result['nll']:.5f} ppl={result['perplexity']:.3f}", flush=True)

    if not args.resume:
        validate()
        save_latest()
    while consumed < budget:
        loader = create_batch_loader(training_data, batch_size=microbatch, training=True,
                                     seed=args.seed, epoch=epoch)
        skip_windows = cursor
        optimizer.zero_grad(set_to_none=True)
        pending_tokens = pending_windows = 0
        pending_loss = 0.0
        update_started = time.perf_counter()
        for raw in loader:
            batch_windows = raw["input_ids"].shape[0]
            if skip_windows >= batch_windows:
                skip_windows -= batch_windows
                continue
            if skip_windows:
                # A saved cursor is always on a microbatch boundary.
                raise ValueError("Checkpoint cursor is not on a microbatch boundary")
            batch = move_batch(raw, device)
            mask = limit_loss_mask_to_budget(batch["loss_mask"], budget - consumed - pending_tokens)
            valid = int(mask.sum())
            if valid == 0:
                break
            model.train()
            logits = model(batch["input_ids"], batch["attention_mask"], batch["position_ids"])
            loss_sum, _ = compute_masked_cross_entropy(logits, batch["target_ids"], mask)
            if not bool(torch.isfinite(loss_sum)):
                raise FloatingPointError("Non-finite training loss")
            loss_sum.backward()
            pending_tokens += valid
            pending_windows += batch_windows
            pending_loss += float(loss_sum.detach())
            cursor += batch_windows
            end_of_epoch = cursor == len(training_data)
            end_of_budget = consumed + pending_tokens == budget
            if pending_windows < effective_batch and not end_of_epoch and not end_of_budget:
                continue
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(pending_tokens)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol["training"]["gradient_clip_global_norm"])
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError("Non-finite gradient norm")
            schedule = protocol["training"]["scheduler"]
            lr = learning_rate(consumed + pending_tokens, budget, args.learning_rate,
                               schedule["warmup_fraction"], schedule["minimum_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            consumed += pending_tokens
            updates += 1
            train_seconds += time.perf_counter() - update_started
            append_event(events, {
                "event": "update", "consumed_tokens": consumed, "updates": updates,
                "epoch": epoch, "window_cursor": cursor, "batch_windows": pending_windows,
                "prediction_tokens": pending_tokens, "training_nll": pending_loss / pending_tokens,
                "learning_rate": lr, "gradient_norm": float(gradient_norm),
                "train_seconds": train_seconds,
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            })
            pending_tokens = pending_windows = 0
            pending_loss = 0.0
            if consumed >= next_validation or consumed == budget:
                while consumed >= next_validation:
                    next_validation += protocol["selection"]["validation_every_prediction_tokens"]
                validate()
            if updates % args.save_every_updates == 0 or consumed == budget or end_of_epoch:
                save_latest()
            if consumed == budget:
                break
            update_started = time.perf_counter()
        if consumed < budget:
            if pending_tokens:
                raise RuntimeError("Epoch ended with unstepped gradients")
            epoch += 1
            cursor = 0
    save_torch(run_dir / "final.pt", {"identity": identity, "model": model.state_dict(),
                                          "consumed_tokens": consumed})
    save_latest()
    summary = {
        "status": "complete", "mode": args.mode, "size": args.size,
        "consumed_tokens": consumed, "updates": updates,
        "best_validation_nll": best_nll, "best_checkpoint_tokens": best_tokens,
        "train_seconds": train_seconds,
        "prediction_tokens_per_second": consumed / train_seconds if train_seconds else None,
        "last_validation_seconds": last_validation_seconds,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "latest_checkpoint_bytes": (run_dir / "latest.pt").stat().st_size,
        "parameters": model.parameter_counts(),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("exploratory", "production"), default="exploratory")
    parser.add_argument("--size", choices=("s", "m", "l"), default="s")
    parser.add_argument("--token-dir", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--snapshot-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--budget-tokens", type=int)
    parser.add_argument("--microbatch-windows", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--save-every-updates", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
