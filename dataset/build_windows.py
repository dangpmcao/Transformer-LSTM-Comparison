"""Create indexes for independent 512-target prediction windows."""

from __future__ import annotations

import argparse
from pathlib import Path

from .common import DEFAULT_DATA_ROOT
from .windows import build_windows as build_window_index, load_windows as verify_window_index

__all__ = ["build_window_index", "verify_window_index"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-dir", type=Path, default=DEFAULT_DATA_ROOT / "processed/tokens", help="Saved token-array folder")
    parser.add_argument("--output-dir", type=Path, help="Window-index folder; default: TOKEN_DIR/windows")
    parser.add_argument("--check", action="store_true", help="Verify existing indexes without rebuilding them")
    args = parser.parse_args()
    result = (verify_window_index(args.output_dir or args.token_dir / "windows", args.token_dir) if args.check
              else build_window_index(args.token_dir, args.output_dir))
    print({split: summary["windows"] for split, summary in result["splits"].items()})


if __name__ == "__main__":
    main()
