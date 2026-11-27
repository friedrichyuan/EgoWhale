"""The shell entry uses the repository YAML and the split scheduler."""
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/wilor.sh"
CONFIG = ROOT / "egowhale/wilor/cfgs/wilor.yaml"


class BatchEntryTests(unittest.TestCase):
    def test_shell_entry_is_unique(self):
        self.assertEqual(list((ROOT / "scripts").glob("*wilor*.sh")), [SCRIPT])

    def test_shell_entry_uses_repository_yaml(self):
        with TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            python.chmod(0o755)
            result = subprocess.run(
                ["bash", str(SCRIPT), "--dry-run"], cwd=directory, check=True,
                capture_output=True, text=True,
                env={**os.environ, "EGOWHALE_PYTHON": str(python), "SPLIT_CONFIG": "/other.yaml"},
            )
        self.assertEqual(result.stdout.splitlines(), [
            "-u", "tools/split_wilor_batch.py", "--config", str(CONFIG),
            "--python", str(python), "--dry-run",
        ])

    def test_shell_entry_rejects_command_line_config(self):
        result = subprocess.run(
            ["bash", str(SCRIPT), "--config=/other.yaml"], capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn(str(CONFIG), result.stderr)
