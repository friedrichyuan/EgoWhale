"""CPU checks of reference parity and the EgoWhale output adapter."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from egowhale.wilor import synced
from egowhale.wilor.step import WiLoR, validate_prediction_input

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT.parent / 'ego2robot_blog'


class SyncedNodeTests(unittest.TestCase):
    @unittest.skipUnless(REFERENCE.is_dir(), 'reference checkout unavailable')
    def test_unmodified_modules_and_base_config_match_reference(self):
        # EgoWhale intentionally extends hand_processor/run_dynhamr with validated
        # smooth fallback; those branches are exercised in test_wilor_smooth_fallback.
        # DA3 sampling and AnyHand candidate extraction are now WiLoR-specific adapters.
        for relative in ('ego2robot/models/wilor_wrapper.py',
                         'ego2robot/models/da3_wrapper.py', 'ego2robot/utils/legacy_mano.py',
                         'ego2robot/utils/progress.py',
                         ):
            with self.subTest(file=relative):
                actual = (synced.SYNC_ROOT / relative).read_text()
                expected = (REFERENCE / relative).read_text()
                self.assertEqual(actual, expected)
        import yaml
        config = yaml.safe_load(synced.CONFIG.read_text())
        local_overrides = ('split', 'resident', 'legacy', 'long_video', 'save_depth_maps', 'min_hand_detection_rate')
        hand_config = {k: v for k, v in config.items() if k not in local_overrides}
        self.assertEqual(hand_config, yaml.safe_load((synced.SYNC_ROOT / 'configs/hand_experiment.yaml').read_text()))
        base = {k: v for k, v in hand_config.items() if not k.startswith('dynhamr_smooth_')}
        self.assertEqual(base, yaml.safe_load((REFERENCE / 'configs/hand_experiment.yaml').read_text()))

    def test_node_publishes_same_points_with_egowhale_coordinate_scale(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / 'input.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10., (80, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((48, 80, 3), dtype=np.uint8))
            writer.release()
            # The presence of this file must not change MP4 routing or be read.
            video.with_suffix('.hdf5').write_bytes(b'not an hdf5 file')
            def fake_runner(command, **kwargs):
                work = Path(command[command.index('--output') + 1])
                hands = work / 'hand_processor'
                hands.mkdir(parents=True, exist_ok=True)
                (work / 'input.json').write_text(json.dumps({'frames': 3, 'fps': 10., 'size': [40, 24]}))
                (work / 'cam_intrinsics.json').write_text(json.dumps({'left': {'fx': 50., 'fy': 55., 'cx': 20., 'cy': 12.}}))
                for side in ('left', 'right'):
                    np.savez(hands / f'hand_data_{side}.npz', hand_detected=np.array([True, False, True]),
                             kpts_3d=np.full((3, 21, 3), .3), kpts_2d=np.full((3, 21, 2), [10., 8.]))
                (work / 'full_completed.json').write_text(json.dumps({'camera_source': 'da3', 'path': 'b',
                    'optimization': {'result_source': 'wilor', 'fallback': False}}))
                (work / 'optimized_hand_points.mp4').write_bytes(b'video')
                self.assertEqual(command[command.index('--path') + 1], 'b')
                self.assertEqual(command[command.index('--camera-source') + 1], 'da3')
                self.assertIn('--video-only', command)
                return subprocess.CompletedProcess(command, 0)
            node = WiLoR()
            with patch('egowhale.wilor.synced.subprocess.run', side_effect=fake_runner):
                node.run(video, root / 'out')
            meta = validate_prediction_input(video, root / 'out')
            self.assertEqual(meta['source'], 'ego2robot_sync')
            self.assertEqual(meta['camera_source'], 'da3')
            self.assertIsNone(meta['calibration_input'])
            with np.load(root / 'out/action/hands_wilor.npz') as result:
                np.testing.assert_allclose(result['joints_camera'][0, 0], .3)
                np.testing.assert_allclose(result['keypoints_2d'][0, 0], np.tile([20., 16.], (21, 1)))
                np.testing.assert_allclose(result['intrinsic'][0, 0], 100.)
                np.testing.assert_array_equal(result['valid'][:, 0], [True, False, True])
                self.assertTrue(np.isnan(result['confidence']).all())
            node.validate_outputs(video, root / 'out')

    def test_hdf5_is_rejected_before_starting_or_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for extension in ('.hdf5', '.h5'):
                source = root / f'episode{extension}'
                source.write_bytes(b'annotated episode')
                output = root / 'out'
                with self.subTest(extension=extension), \
                     patch('egowhale.wilor.synced.subprocess.run') as runner:
                    with self.assertRaisesRegex(ValueError, 'Retarget'):
                        WiLoR().run(source, output)
                    with self.assertRaisesRegex(ValueError, 'Retarget'):
                        synced.run(source, output, synced.CONFIG, Path('python'), 'auto', None, .3)
                    runner.assert_not_called()
                    self.assertFalse(output.exists())

    def test_both_cli_entries_reject_hdf5_and_path_a_without_models(self):
        import sys
        for entry in ([sys.executable, '-m', 'egowhale.wilor'],
                      [sys.executable, str(synced.RUNNER)]):
            with self.subTest(entry=entry):
                result = subprocess.run([*entry, '--video', 'episode.hdf5', '--output', '/unused'],
                                        cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('Retarget', result.stderr)
        result = subprocess.run([sys.executable, str(synced.RUNNER), '--path', 'a'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('invalid choice', result.stderr)


if __name__ == '__main__':
    unittest.main()
