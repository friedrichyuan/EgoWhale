"""Batch orchestration without GPU models; verifies isolation and output collection."""
import contextlib
import csv
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('wilor_sync_batch', ROOT / 'tools/batch_wilor_ego2robot.py')
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class BatchTests(unittest.TestCase):
    def test_multigpu_jobs_collect_videos_and_keep_failures_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'data' / 'add_remove_lid'
            inputs.mkdir(parents=True)
            for i in range(3):
                (inputs / f'{i}.mp4').write_bytes(b'fake input')
            fake = root / 'fake_node.py'
            fake.write_text('''import argparse,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');p.add_argument('--video');p.add_argument('--output');p.add_argument('--camera-source');a,_=p.parse_known_args()
assert a.camera_source == 'da3'
if a.check: assert '--video-only' in sys.argv and sys.argv[sys.argv.index('--path') + 1] == 'b'; print('preflight ok'); sys.exit(0)
if Path(a.video).stem == '1': sys.exit(7)
out=Path(a.output); work=out/'action/wilor_ego2robot'; work.mkdir(parents=True)
(work/'optimized_hand_points.mp4').write_bytes(b'video')
(work/'full_completed.json').write_text('{}')
(work/'annotation_counts.json').write_text(json.dumps({'left': 2 if Path(a.video).stem == '0' else 0, 'right':0}))
(out/'action/hands_wilor.npz').write_bytes(b'data')
print('mock video completed on GPU',os.environ['CUDA_VISIBLE_DEVICES'])
''')
            args = ['batch', '--input', str(inputs), '--output-root', str(root / 'outputs'),
                    '--gpu-ids', '0,1', '--python', sys.executable, '--heartbeat', '.1']
            with patch.object(batch, 'NODE', fake), patch.object(batch, 'PREFLIGHT', fake), \
                 patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(batch.main(), 1)
            run = next((root / 'outputs').glob('batch_*'))
            with (run / 'results.csv').open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            by_name = {Path(row['source']).name: row for row in rows}
            self.assertEqual(by_name['0.mp4']['gpu'], '0')
            self.assertEqual(by_name['1.mp4']['gpu'], '1')
            self.assertTrue(by_name['1.mp4']['status'].startswith('失败'))
            self.assertIn('无可见', by_name['2.mp4']['status'])
            self.assertEqual(len(list((run / 'final_videos').glob('*.mp4'))), 2)
            self.assertTrue(all(p.resolve().is_file() for p in (run / 'final_videos').glob('*.mp4')))
            self.assertTrue(all(Path(row['log']).is_file() for row in rows))


if __name__ == '__main__':
    unittest.main()
