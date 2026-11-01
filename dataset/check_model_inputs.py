"""Check one real article through tokenization, batching, and masked loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tokenizer.core import DEFAULT_OUTPUT as DEFAULT_TOKENIZER_DIR
from .check_inputs import check_inputs as check_model_inputs
from .common import DEFAULT_DATA_ROOT, write_json

__all__ = ["check_model_inputs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="External corpus folder")
    parser.add_argument("--token-dir", type=Path, help="Token-array folder; default: DATA_ROOT/processed/tokens")
    parser.add_argument("--window-dir", type=Path, help="Window-index folder; default: TOKEN_DIR/windows")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR, help="Trained tokenizer bundle")
    parser.add_argument("--report-file", type=Path, help="Check report; default: TOKEN_DIR/checks.json")
    args = parser.parse_args()
    torch.set_num_threads(2)
    token_dir = args.token_dir or args.data_root / "processed/tokens"
    result = check_model_inputs(args.data_root, token_dir, args.tokenizer_dir, args.window_dir)
    write_json(args.report_file or token_dir / "checks.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
