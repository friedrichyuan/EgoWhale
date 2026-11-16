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
    def test_hand_implementation_and_config_match_reference(self):
        for relative in ('ego2robot/processors/hand_processor.py', 'ego2robot/processors/depth_pose_processor.py',
                         'ego2robot/models/anyhand_wrapper.py', 'ego2robot/models/wilor_wrapper.py',
                         'ego2robot/models/da3_wrapper.py', 'ego2robot/utils/legacy_mano.py',
                         'ego2robot/utils/progress.py', 'ego2robot/utils/egodex_path_a.py',
                         'scripts/run_dynhamr.py', 'configs/hand_experiment.yaml'):
            with self.subTest(file=relative):
                self.assertEqual((synced.SYNC_ROOT / relative).read_bytes(), (REFERENCE / relative).read_bytes())
        self.assertEqual(synced.CONFIG.read_bytes(), (REFERENCE / 'configs/hand_experiment.yaml').read_bytes())

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

    def test_hdf5_uses_path_a_and_publishes_wilor_joint_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, h5 = root / 'episode.mp4', root / 'episode.hdf5'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10., (80, 48))
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((48, 80, 3), dtype=np.uint8))
            writer.release()
            h5.write_bytes(b'fake HDF5 read only by mocked worker')
            def fake_runner(command, **kwargs):
                self.assertEqual(command[command.index('--video') + 1], str(h5))
                self.assertEqual(command[command.index('--path') + 1], 'a')
                self.assertEqual(command[command.index('--camera-source') + 1], 'auto')
                self.assertNotIn('--video-only', command)
                work = Path(command[command.index('--output') + 1]); hands = work / 'hand_processor'
                hands.mkdir(parents=True, exist_ok=True)
                (work / 'input.json').write_text(json.dumps({'frames': 1, 'fps': 10., 'size': [80, 48],
                    'request': {'processing_path': 'a'}}))
                (work / 'cam_intrinsics.json').write_text(json.dumps({'left': {'fx': 50., 'fy': 50., 'cx': 40., 'cy': 24.}}))
                xyz = np.ones((1, 21, 3), dtype=np.float32); xyz[0, :, 0] = np.arange(21)
                for side in ('left', 'right'):
                    np.savez(hands / f'hand_data_{side}.npz', hand_detected=np.array([True]),
                             kpts_3d=xyz, kpts_2d=np.ones((1, 21, 2)))
                (work / 'full_completed.json').write_text(json.dumps({'path': 'a', 'camera_source': 'egodex',
                    'optimization': {'result_source': 'egodex_annotations', 'fallback': False}}))
                return subprocess.CompletedProcess(command, 0)
            node = WiLoR()
            with patch('egowhale.wilor.synced.subprocess.run', side_effect=fake_runner):
                node.run(h5, root / 'out')
            with np.load(root / 'out/action/hands_wilor.npz') as result:
                np.testing.assert_array_equal(result['joints_camera'][0, 0, :, 0],
                    [0, 17, 18, 19, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            meta = validate_prediction_input(h5, root / 'out')
            self.assertEqual(meta['processing_path'], 'a')
            self.assertEqual(meta['optimization']['result_source'], 'egodex_annotations')


if __name__ == '__main__':
    unittest.main()
