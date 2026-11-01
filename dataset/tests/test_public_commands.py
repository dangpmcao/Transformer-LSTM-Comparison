"""Exercise readable public commands with portable paths and real CLI parsing."""

import os
import subprocess
import sys

from dataset.common import read_json
from dataset.model_inputs import NextTokenWindowDataset, create_batch_loader
from tokenizer.tests.test_tokenizer import fixture_data, bundle


def run_command(module, *arguments):
    return subprocess.run(
        [sys.executable, "-B", "-m", module, *map(str, arguments)],
        text=True, capture_output=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false"),
    )


def test_readable_commands_build_and_check_custom_paths(bundle, tmp_path):
    data_root, _, tokenizer_dir = bundle
    token_dir = tmp_path / "token arrays"
    window_dir = tmp_path / "prediction windows"
    report_file = tmp_path / "input check.json"
    commands = [
        ("dataset.encode_articles", "--data-root", data_root, "--tokenizer-dir", tokenizer_dir,
         "--output-dir", token_dir),
        ("dataset.build_windows", "--token-dir", token_dir, "--output-dir", window_dir),
        ("dataset.check_model_inputs", "--data-root", data_root, "--token-dir", token_dir,
         "--window-dir", window_dir, "--tokenizer-dir", tokenizer_dir, "--report-file", report_file),
        ("dataset.encode_articles", "--output-dir", token_dir, "--check"),
        ("dataset.build_windows", "--token-dir", token_dir, "--output-dir", window_dir, "--check"),
    ]
    for module, *arguments in commands:
        result = run_command(module, *arguments)
        assert result.returncode == 0, result.stderr
    assert read_json(report_file)["status"] == "PASS"
    assert not (token_dir / "checks.json").exists()
    windows = NextTokenWindowDataset(token_dir, "train", window_dir)
    batch = next(iter(create_batch_loader(windows, 2, training=True, seed=13, epoch=0)))
    assert batch["input_ids"].shape == (2, 512)
    assert batch["input_ids"][:, 0].eq(1).all()
    assert not batch["target_ids"][batch["loss_mask"]].eq(0).any()
