"""Exercise the persistent worker protocol and crash isolation without GPU models."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('resident_batch', ROOT / 'tools/batch_wilor_ego2robot.py')
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class ResidentClientTests(unittest.TestCase):
    def test_reuses_process_and_restarts_after_failed_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / 'worker.py'
            fake.write_text('''import json, os, sys
for line in sys.stdin:
    task=json.loads(line)
    if task.get('command') == 'stop': break
    print('worker-pid', os.getpid(), flush=True)
    bad=task.get('fail', False)
    print('@@EGOWHALE_WILOR_RESPONSE@@'+json.dumps({'ok':not bad,'error':'injected failure'}), flush=True)
    if bad: sys.exit(1)
''')
            active, lock, stop = set(), threading.Lock(), threading.Event()
            with patch.object(batch, 'RESIDENT_NODE', fake):
                client = batch.ResidentClient(Path(sys.executable), os.environ.copy(), .1, active, lock, stop)
                try:
                    def submit(name, fail=False):
                        return client.request({'fail': fail}, root / f'{name}.log', name, root / 'progress')
                    self.assertEqual(submit('one'), 0)
                    first_pid = client.process.pid
                    self.assertEqual(submit('two'), 0)
                    self.assertEqual(client.process.pid, first_pid)
                    self.assertEqual(submit('failure', True), 1)
                    self.assertIsNone(client.process)
                    self.assertEqual(submit('after'), 0)
                    self.assertNotEqual(client.process.pid, first_pid)
                finally:
                    client.close()
            self.assertFalse(active)
            self.assertIn(str(first_pid), (root / 'one.log').read_text())
            self.assertIn(str(first_pid), (root / 'two.log').read_text())


if __name__ == '__main__':
    unittest.main()
