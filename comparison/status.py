"""One-line-per-run progress table for training runs (reads run.json / metrics.jsonl / summary.json).

    python -m comparison.status                         # lstm/runs and transformer/runs
    python -m comparison.status --watch 60              # refresh every 60 s
    python -m comparison.status --roots lstm/runs/tuning

States: running, done, stalled (no update for --stall-minutes), not started.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [PROJECT_ROOT / "lstm/runs", PROJECT_ROOT / "transformer/runs"]


def _tokens(value: float | None) -> str:
    return "-" if value is None else f"{value / 1e6:.1f}M"


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0:
        return "-"
    hours, rest = divmod(int(seconds), 3600)
    return f"{hours}h{rest // 60:02d}m" if hours else f"{rest // 60}m{rest % 60:02d}s"


def run_status(run_dir: Path, now: datetime, stall_minutes: float, recent_updates: int = 20) -> dict:
    identity = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))["identity"]
    budget = identity["budget_tokens"]
    updates, best, last_validation = [], None, None
    metrics = run_dir / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["event"] == "update":
                updates.append(row)
            elif row["event"] == "validation":
                last_validation = row["nll"]
                best = row["nll"] if best is None else min(best, row["nll"])
    last = updates[-1] if updates else None
    consumed = last["consumed_tokens"] if last else 0
    rate = None
    if len(updates) > 1:  # Recent throughput from training-step time only.
        window = updates[-recent_updates:]
        seconds = window[-1]["train_seconds"] - window[0]["train_seconds"]
        rate = (window[-1]["consumed_tokens"] - window[0]["consumed_tokens"]) / seconds if seconds > 0 else None
    if (run_dir / "summary.json").exists():
        state = "done"
    elif last is None:
        state = "not started"
    else:
        idle = (now - datetime.fromisoformat(last["at"])).total_seconds() / 60
        state = f"stalled {idle:.0f}m" if idle > stall_minutes else "running"
    return {
        "run": run_dir, "architecture": identity.get("architecture", "transformer"), "state": state,
        "consumed": consumed, "budget": budget, "percent": 100 * consumed / budget if budget else 0,
        "train_nll": last["training_nll"] if last else None, "val_nll": last_validation, "best_val": best,
        "rate": rate, "eta": (budget - consumed) / rate if rate and state == "running" else None,
        "lr": identity.get("learning_rate"), "dropout": identity.get("dropout", identity["model"].get("dropout")),
    }


def table(roots: list[Path], stall_minutes: float) -> str:
    now = datetime.now(timezone.utc)
    rows = [run_status(path.parent, now, stall_minutes)
            for root in roots if root.exists() for path in sorted(root.rglob("run.json"))]
    if not rows:
        return "No runs found under: " + ", ".join(map(str, roots))
    header = f"{'run':44} {'state':11} {'tokens':>16} {'%':>4} {'lr':>7} {'drop':>4} {'train':>6} {'val':>6} {'best':>6} {'tok/s':>7} {'ETA':>7}"
    lines = [header, "-" * len(header)]
    for r in rows:
        name = str(r["run"].relative_to(PROJECT_ROOT) if r["run"].is_relative_to(PROJECT_ROOT) else r["run"])
        fmt = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        rate = "-" if r["rate"] is None else f"{r['rate'] / 1e3:.1f}k"
        tokens = f"{_tokens(r['consumed'])} / {_tokens(r['budget'])}"
        lines.append(
            f"{name[-44:]:44} {r['state']:11} {tokens:>16} {r['percent']:4.0f} {r['lr'] or 0:7.0e} "
            f"{r['dropout'] or 0:4.1f} {fmt(r['train_nll']):>6} {fmt(r['val_nll']):>6} "
            f"{fmt(r['best_val']):>6} {rate:>7} {_duration(r['eta']):>7}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+", default=DEFAULT_ROOTS)
    parser.add_argument("--watch", type=float, metavar="SECONDS", help="Refresh until interrupted")
    parser.add_argument("--stall-minutes", type=float, default=10.0)
    args = parser.parse_args()
    while True:
        output = table([root.resolve() for root in args.roots], args.stall_minutes)
        if not args.watch:
            print(output)
            return
        print("\033[2J\033[H" + datetime.now().strftime("%H:%M:%S") + "\n" + output, flush=True)
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
