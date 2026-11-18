"""The old batch entry point must delegate to the split scheduler."""
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

class BatchEntryTests(unittest.TestCase):
    def test_old_batch_entry_points_delegate_to_split(self):
        for name in ("run_wilor_resident_multigpu.sh", "run_wilor_ego2robot_batch.sh",
                     "run_egodex_fixed500_wilor_multigpu.sh", "run_wilor_batch.sh"):
            text = (ROOT / "scripts" / name).read_text()
            self.assertIn("run_wilor_split_multigpu.sh", text)
            self.assertNotIn("tools/batch_wilor_ego2robot.py", text)
