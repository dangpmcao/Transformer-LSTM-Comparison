"""Readable settings commands preserve immutable snapshots and readiness gates."""

import os
import subprocess
import sys

from dataset.common import read_json, sha256_file
from comparison.tests.test_protocol import production_shape_fixture


def test_settings_command_preserves_prepared_readiness_gate(production_shape_fixture, tmp_path):
    tokenizer_dir, token_dir, settings_file = production_shape_fixture
    snapshot_file = tmp_path / "prepared settings.json"
    command = [sys.executable, "-B", "-m", "comparison.experiment_settings",
               "--token-dir", str(token_dir), "--tokenizer-dir", str(tokenizer_dir),
               "--settings-file", str(settings_file), "--snapshot-file", str(snapshot_file)]
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    created = subprocess.run(command, text=True, capture_output=True, env=environment)
    assert created.returncode == 0, created.stderr
    assert read_json(snapshot_file)["production_ready"] is False
    before = sha256_file(snapshot_file)
    checked = subprocess.run([*command, "--check"], text=True, capture_output=True, env=environment)
    assert checked.returncode == 0, checked.stderr
    blocked = subprocess.run([*command, "--check", "--require-ready"], text=True, capture_output=True, env=environment)
    assert blocked.returncode != 0 and "Production is blocked" in blocked.stderr
    overwrite = subprocess.run(command, text=True, capture_output=True, env=environment)
    assert overwrite.returncode != 0 and "Refusing to overwrite" in overwrite.stderr
    assert sha256_file(snapshot_file) == before
