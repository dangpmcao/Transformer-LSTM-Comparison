"""Tokenizer contract shared by training, verification, and future model pipelines.

Special IDs are inserted explicitly at article/window boundaries by the caller.
Ordinary text, including '<bos>', always goes through byte-level BPE as text.
"""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers, trainers

from dataset.common import PROJECT_ROOT, SOURCE_LOCK, contained_path, read_json, sha256_file

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = PROJECT_ROOT / "tokenizer/artifacts"
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
SPECIAL_IDS = dict(zip(SPECIAL_TOKENS, range(4)))
FROZEN_SETTINGS = {
    "schema_version": 1, "algorithm": "byte_level_bpe", "tokenizers_version": "0.22.1",
    "special_tokens": list(SPECIAL_TOKENS), "initial_alphabet": "full_byte_level",
    "normalization": None, "add_prefix_space": False, "use_regex": True,
    "bpe_dropout": None, "automatic_special_tokens": False, "parallelism": False,
}
PROBES = (
    "", " ", "  leading and trailing  ", "\t\n\r\n", "Hello, World!",
    "Café cafe\u0301 Tiếng Việt", "日本語 中文 한국어", "😀🫀🚀", "\x00\x01\x7f",
    "<pad><bos><eos><unk>", "literal <bos> is text; <unk> too",
    " = Heading = \n = = Subsection = = \n", "".join(chr(i) for i in range(256)),
)


def validate_config(config: dict[str, Any]) -> None:
    tunable = {"vocab_size", "min_frequency", "batch_size"}
    if set(config) != set(FROZEN_SETTINGS) | tunable:
        raise ValueError("Configuration keys must match tokenizer/config.json")
    # Only these numerical choices are ablatable; text/control semantics stay fixed.
    for key, expected in FROZEN_SETTINGS.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"Unsupported tokenizer contract: {key}={config[key]!r}")
    for key, minimum in [("vocab_size", 260), ("min_frequency", 1), ("batch_size", 1)]:
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    installed = importlib.metadata.version("tokenizers")
    if installed != config["tokenizers_version"]:
        raise ValueError(f"Expected tokenizers=={config['tokenizers_version']}, installed {installed}")


def build_tokenizer(config: dict[str, Any]) -> tuple[Tokenizer, trainers.BpeTrainer]:
    validate_config(config)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>", dropout=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    # No normalizer, postprocessor, automatic BOS/EOS, truncation, or padding.
    tokenizer.encode_special_tokens = True
    trainer = trainers.BpeTrainer(
        vocab_size=config["vocab_size"], min_frequency=config["min_frequency"],
        special_tokens=[AddedToken(token, normalized=False, special=True) for token in SPECIAL_TOKENS],
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()), show_progress=True,
    )
    return tokenizer, trainer


def verify_structure(tokenizer: Tokenizer, config: dict[str, Any]) -> None:
    validate_config(config)
    if tokenizer.get_vocab_size() != config["vocab_size"]:
        raise ValueError(f"Expected exactly {config['vocab_size']} IDs, got {tokenizer.get_vocab_size()}")
    if set(tokenizer.get_vocab().values()) != set(range(config["vocab_size"])):
        raise ValueError("Vocabulary IDs are not contiguous")
    for token, token_id in SPECIAL_IDS.items():
        if tokenizer.token_to_id(token) != token_id:
            raise ValueError(f"Incorrect reserved ID for {token}")
    if any(tokenizer.token_to_id(char) is None for char in pre_tokenizers.ByteLevel.alphabet()):
        raise ValueError("The complete 256-byte alphabet is required")
    spec = json.loads(tokenizer.to_str())
    for key in ["normalizer", "post_processor", "padding", "truncation"]:
        if spec[key] is not None:
            raise ValueError(f"Unexpected {key}")
    pre = spec["pre_tokenizer"]
    if pre["type"] != "ByteLevel" or pre["add_prefix_space"] or not pre["use_regex"]:
        raise ValueError("Incorrect byte-level pre-tokenization")
    if spec["decoder"]["type"] != "ByteLevel" or spec["model"]["type"] != "BPE" or spec["model"]["dropout"] is not None:
        raise ValueError("Incorrect decoder/model or enabled BPE dropout")
    if not tokenizer.encode_special_tokens:
        raise ValueError("Literal special-token strings must be encoded as ordinary text")


def load_tokenizer(artifacts: Path = DEFAULT_OUTPUT, *, require_checks: bool = True) -> Tokenizer:
    """Load a frozen artifact and restore the nonserialized literal-text setting.

    HF Tokenizer.from_file alone resets encode_special_tokens to False in 0.22.1.
    Future model/data code should use this loader instead of the bare HF loader.
    Only the verification command should disable the completed-check gate.
    """
    artifacts = Path(artifacts)
    manifest = read_json(artifacts / "manifest.json")
    if set(manifest["files"]) != {"tokenizer.json", "vocab.json", "merges.txt", "config.json", "special_tokens.json"}:
        raise ValueError("Incomplete tokenizer artifact manifest")
    for name, record in manifest["files"].items():
        if record["path"] != name:
            raise ValueError(f"Incorrect tokenizer artifact path for {name}")
        path = contained_path(artifacts, record["path"])
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"Changed tokenizer artifact: {record['path']}")
    if sha256_file(artifacts / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer fingerprint differs from the frozen manifest")
    config = read_json(artifacts / "config.json")
    if (manifest["config"] != config or manifest["vocab_size"] != config["vocab_size"]
            or manifest["special_token_ids"] != SPECIAL_IDS
            or read_json(artifacts / "special_tokens.json") != SPECIAL_IDS):
        raise ValueError("Inconsistent tokenizer configuration/special-token metadata")
    tokenizer = Tokenizer.from_file(str(artifacts / "tokenizer.json"))
    tokenizer.encode_special_tokens = True
    verify_structure(tokenizer, config)
    # The standalone exports must describe the same BPE as tokenizer.json.
    exported_model = models.BPE.from_file(
        str(artifacts / "vocab.json"), str(artifacts / "merges.txt"), unk_token="<unk>",
    )
    exported = json.loads(Tokenizer(exported_model).to_str())["model"]
    model = json.loads(tokenizer.to_str())["model"]
    if exported["vocab"] != model["vocab"] or exported["merges"] != model["merges"]:
        raise ValueError("Standalone vocabulary/merges disagree with tokenizer.json")
    if require_checks:
        checks_path = artifacts / "checks.json"
        statistics_path = artifacts / "statistics.csv"
        if not checks_path.is_file() or not statistics_path.is_file():
            raise ValueError("Missing tokenizer verification; run python -m tokenizer.check")
        checks = read_json(checks_path)
        if (checks.get("status") != "PASS"
                or checks.get("manifest_sha256") != sha256_file(artifacts / "manifest.json")
                or checks.get("tokenizer_sha256") != manifest["tokenizer_sha256"]
                or checks.get("statistics_sha256") != sha256_file(statistics_path)):
            raise ValueError("Stale tokenizer verification; run python -m tokenizer.check")
    return tokenizer


def encode_text(tokenizer: Tokenizer, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
    """Encode content; only explicit caller flags introduce control IDs."""
    if not tokenizer.encode_special_tokens:
        raise ValueError("Use tokenizer.core.load_tokenizer before encoding text")
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    if any(token_id in SPECIAL_IDS.values() for token_id in ids):
        raise ValueError("A control ID appeared in ordinary text")
    return ([SPECIAL_IDS["<bos>"]] if add_bos else []) + ids + ([SPECIAL_IDS["<eos>"]] if add_eos else [])


def article_batches(path: Path, split: str, expected_count: int, batch_size: int) -> Iterator[list[str]]:
    """Stream every article once, in its persisted order, without sampling."""
    batch: list[str] = []
    count = 0
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            article = json.loads(line)
            if article["split"] != split or article["article_id"] != f"{split}:{index:06d}":
                raise ValueError(f"Wrong split/order in {path.name}, line {index + 1}")
            text = article["text"]
            if not isinstance(text, str):
                raise ValueError(f"Non-string article text: {article['article_id']}")
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != article["text_sha256"]:
                raise ValueError(f"Changed article text: {article['article_id']}")
            batch.append(text)
            count += 1
            if len(batch) == batch_size:
                yield batch
                batch = []
    if count != expected_count:
        raise ValueError(f"Expected {expected_count} {split} articles, got {count}")
    if batch:
        yield batch


def verified_data_manifest(root: Path) -> dict[str, Any]:
    manifest = read_json(root / "manifest.json")
    validation = read_json(root / "validation.json")
    if validation["status"] != "PASS" or validation["manifest_sha256"] != sha256_file(root / "manifest.json"):
        raise ValueError("Stage 1 data validation is missing or stale; run python -m dataset.validate")
    source = manifest["sources"]["wikitext"]
    if source != read_json(SOURCE_LOCK)["wikitext"]:
        raise ValueError("Expected the source-locked raw WikiText-103 configuration")
    return manifest


def data_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Bind corpus content and split structure independently of host/timestamps."""
    return {
        "source": manifest["sources"]["wikitext"],
        "splits": {
            split: {key: manifest["wikitext_splits"][split][key]
                    for key in ["file", "articles", "text_bytes"]}
            for split in ["train", "validation", "test"]
        },
    }


def verified_article_path(root: Path, manifest: dict[str, Any], split: str) -> Path:
    record = manifest["wikitext_splits"][split]["file"]
    path = contained_path(root, record["path"])
    if path != (root / "processed/articles" / f"{split}.jsonl").resolve():
        raise ValueError(f"Unexpected {split} article path")
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"Changed {split} article file; rerun Stage 1 verification")
    return path
