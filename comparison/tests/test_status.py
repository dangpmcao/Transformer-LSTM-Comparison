"""Progress table: running/done/stalled states, throughput and ETA from metrics.jsonl."""

import json
from datetime import datetime, timedelta, timezone

from comparison.status import run_status, table


def make_run(root, name, *, done=False, minutes_ago=1.0):
    run = root / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"identity": {
        "budget_tokens": 1000, "architecture": "lstm", "learning_rate": 1e-3, "model": {"dropout": 0.1}}}))
    at = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    rows = [{"event": "update", "at": at.isoformat(), "consumed_tokens": 100 * i, "training_nll": 5.0 - i / 10,
             "train_seconds": float(i)} for i in (1, 2, 3)]
    rows.append({"event": "validation", "at": at.isoformat(), "consumed_tokens": 300, "nll": 4.5})
    (run / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if done:
        (run / "summary.json").write_text("{}")
    return run


def test_states_rate_and_eta(tmp_path):
    now = datetime.now(timezone.utc)
    running = run_status(make_run(tmp_path, "a"), now, stall_minutes=10)
    assert running["state"] == "running" and running["consumed"] == 300 and running["percent"] == 30
    assert running["rate"] == 100 and running["eta"] == 7 and running["best_val"] == 4.5
    assert run_status(make_run(tmp_path, "b", minutes_ago=30), now, 10)["state"].startswith("stalled")
    assert run_status(make_run(tmp_path, "c", done=True), now, 10)["state"] == "done"
    text = table([tmp_path], 10)
    assert "running" in text and "done" in text and "stalled" in text
