"""Exercise split queues, GPU roles, failure isolation, and accounting."""
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
spec = importlib.util.spec_from_file_location('split_batch', ROOT / 'tools/split_wilor_batch.py')
split = importlib.util.module_from_spec(spec)
spec.loader.exec_module(split)


class FakeResident:
    def __init__(self, python, env, interval, active, lock, stop):
        self.gpu = env['CUDA_VISIBLE_DEVICES']

    def request(self, request, log, label, progress):
        log.write_text('mock worker')
        source = Path(request['video']); output = Path(request['output'])
        if request['stage'] == 'camera':
            assert self.gpu in ('0', '1')
            if source.stem == '1':
                return 1
            return 0
        assert self.gpu == '2'
        if source.stem == '2':
            return 1
        work = output / 'action/wilor_ego2robot'
        work.mkdir(parents=True)
        (work / 'optimized_hand_points.mp4').write_bytes(b'video')
        (work / 'input.json').write_text(json.dumps({'frames': 10}))
        (work / 'annotation_counts.json').write_text(json.dumps({'left': 5, 'right': 0}))
        (output / 'action/hands_wilor.npz').write_bytes(b'hand')
        return 0

    def close(self):
        pass


class SplitBatchTests(unittest.TestCase):
    def test_queues_preserve_gpu_roles_and_record_partial_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); inputs = root / 'input'; inputs.mkdir()
            for i in range(4): (inputs / f'{i}.mp4').write_bytes(b'video')
            cfg = root / 'config.yaml'
            cfg.write_text(f'''dynhamr_enabled: false
split:
  input_dir: {inputs}
  output_root: {root / 'outputs'}
  camera_gpus: [0, 1]
  wilor_gpus: [2]
  max_videos: 0
  max_pending_camera: 1
  heartbeat_seconds: 0.1
''')
            argv = ['split', '--config', str(cfg), '--python', sys.executable]
            with patch.object(split, 'ResidentClient', FakeResident), patch.object(sys, 'argv', argv), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(split.main(), 1)
            output = next((root / 'outputs').glob('split_*'))
            with (output / 'results.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = {Path(row['source']).stem: row for row in csv.DictReader(handle)}
            self.assertEqual(set(rows), {'0', '1', '2', '3'})
            self.assertEqual(rows['1']['status'], 'DA3失败')
            self.assertEqual(rows['2']['status'], 'WiLoR失败')
            self.assertEqual(rows['0']['status'], '成功')
            self.assertEqual(rows['3']['status'], '成功')
            self.assertEqual(rows['0']['wilor_gpu'], '2')
            self.assertIn(rows['0']['camera_gpu'], ('0', '1'))
            summary = json.loads((output / 'throughput.json').read_text())
            self.assertEqual((summary['success'], summary['failed_or_interrupted'], summary['successful_frames']), (2, 2, 20))
            self.assertEqual(len(list((output / 'final_videos').glob('*.mp4'))), 2)


if __name__ == '__main__':
    unittest.main()
