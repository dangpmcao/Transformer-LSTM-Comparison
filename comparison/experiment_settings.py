"""Save or verify experiment settings together with their exact input identities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.common import DEFAULT_DATA_ROOT
from tokenizer.core import DEFAULT_OUTPUT as DEFAULT_TOKENIZER_DIR
from .protocol import (
    DEFAULT_LOCK as DEFAULT_SNAPSHOT_FILE,
    DEFAULT_PROTOCOL as DEFAULT_SETTINGS_FILE,
    check_lock as verify_settings_snapshot,
    freeze as save_settings_snapshot,
    load_protocol as load_experiment_settings,
)

__all__ = ["load_experiment_settings", "save_settings_snapshot", "verify_settings_snapshot"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-dir", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens", help="Saved token-array folder")
    parser.add_argument("--window-dir", type=Path, help="Window-index folder; default: TOKEN_DIR/windows")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR, help="Trained tokenizer bundle")
    parser.add_argument("--settings-file", type=Path, default=DEFAULT_SETTINGS_FILE, help="Editable experiment rules (YAML)")
    parser.add_argument("--snapshot-file", type=Path, default=DEFAULT_SNAPSHOT_FILE, help="Saved rules and input hashes (JSON)")
    parser.add_argument("--pilot-report", type=Path, help="Measured timing-trial evidence for a new training-ready snapshot")
    parser.add_argument("--check", action="store_true", help="Verify an existing snapshot without replacing it")
    parser.add_argument("--require-ready", action="store_true", help="Also require measured timing evidence and exact budgets")
    args = parser.parse_args()
    if args.require_ready and not args.check:
        parser.error("--require-ready requires --check")
    if args.check and args.pilot_report:
        parser.error("--pilot-report is used when saving a new snapshot")
    if args.check:
        result = verify_settings_snapshot(
            args.snapshot_file, args.token_dir, args.window_dir,
            protocol=args.settings_file, artifacts=args.tokenizer_dir, require_production=args.require_ready,
        )
    else:
        result = save_settings_snapshot(
            args.token_dir, args.window_dir, artifacts=args.tokenizer_dir,
            protocol=args.settings_file, output=args.snapshot_file, pilot_path=args.pilot_report,
        )
    print(json.dumps({"production_ready": result["production_ready"], "snapshot_file": str(args.snapshot_file)}))


if __name__ == "__main__":
    main()
