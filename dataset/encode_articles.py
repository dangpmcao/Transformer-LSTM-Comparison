"""Convert complete WikiText articles into saved token ID arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

from tokenizer.core import DEFAULT_OUTPUT as DEFAULT_TOKENIZER_DIR
from .common import DEFAULT_DATA_ROOT
from .encode import encode as encode_articles, load_cache as verify_token_arrays

__all__ = ["encode_articles", "verify_token_arrays"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="External corpus folder")
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR, help="Trained tokenizer bundle")
    parser.add_argument("--output-dir", type=Path, help="Token-array folder; default: DATA_ROOT/processed/tokens")
    parser.add_argument("--check", action="store_true", help="Verify existing arrays without rebuilding them")
    args = parser.parse_args()
    result = (verify_token_arrays(args.output_dir or args.data_root / "processed/tokens") if args.check
              else encode_articles(args.data_root, args.tokenizer_dir, args.output_dir))
    print(f"Token arrays: {result['status']}")


if __name__ == "__main__":
    main()
