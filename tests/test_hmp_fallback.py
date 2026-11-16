"""HMP failure still publishes a playable prediction with truthful metadata."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import av
import cv2
import numpy as np

from egowhale.paths import HANDS_WILOR
from egowhale.wilor.pipeline import refine
from egowhale.wilor.pipeline.contracts import Artifact, derive, write_artifact
from egowhale.wilor.pipeline.runner import DEFAULTS, model_identity, publish_prediction
from egowhale.wilor.pipeline.workers.dynhamr import attempt_prior
from tools.run_wilor_video import execute, validate_full_prediction


def tracks_and_camera(count=8):
    valid = np.ones((count, 2), bool)
    intrinsic = np.array([[100., 0., 48.], [0., 100., 32.], [0., 0., 1.]])
    arrays = {'valid': valid, 'frame_index': np.arange(count), 'intrinsic': intrinsic, 'fps': np.array(30.),
              'joints_camera': np.full((count, 2, 21, 3), .3),
              'keypoints_2d': np.full((count, 2, 21, 2), [30., 30.]),
              'global_orient': np.tile(np.eye(3), (count, 2, 1, 1)),
              'hand_pose': np.tile(np.eye(3), (count, 2, 15, 1, 1)),
              'translation': np.full((count, 2, 3), .3), 'betas': np.zeros((count, 2, 10)),
              'confidence': np.ones((count, 2)), 'observed': valid.copy(),
              'interpolated': np.zeros_like(valid), 'reasons': np.zeros(valid.shape, np.uint16)}
    tracks = Artifact(arrays, {'kind': 'filtered'})
    camera = Artifact({'frame_index': np.arange(count), 'intrinsic': intrinsic,
                       'w2c': np.tile(np.eye(4), (count, 1, 1)), 'fps': np.array(30.)},
                      {'kind': 'camera', 'width': 96, 'height': 64})
    return tracks, camera


class HMPFallbackTests(unittest.TestCase):
    def test_prior_exception_returns_a_nonfatal_worker_response(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stderr(io.StringIO()):
                response = attempt_prior(Mock(side_effect=AssertionError('expected 128 frames')), Path(directory))
            self.assertTrue(response['ok'])
            self.assertTrue(response['optimizer_fallback'])
            self.assertFalse(response['prior_executed'])
            self.assertEqual(json.loads((Path(directory) / 'hmp_fallback.json').read_text()), response)

    def test_successful_prior_does_not_activate_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(attempt_prior(Mock(return_value='prior result'), Path(directory)))
            self.assertFalse((Path(directory) / 'hmp_fallback.json').exists())

    def test_hmp_timeout_falls_back_but_root_failure_stays_an_error(self):
        for phase in ('hmp', 'root'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                tracks, camera = tracks_and_camera()
                options = {**DEFAULTS['refine'], 'workspace': directory, 'camera_mode': 'author-static',
                           'mano': str(Path(directory) / 'MANO_RIGHT.pkl')}
                def timeout(payload):
                    (Path(directory) / 'optimizer_stage.json').write_text(json.dumps({'stage': phase}))
                    raise TimeoutError('worker timed out')
                client = Mock()
                client.request.side_effect = timeout
                with patch.object(refine, 'WorkerClient', return_value=client), contextlib.redirect_stdout(io.StringIO()):
                    if phase == 'root':
                        with self.assertRaises(TimeoutError):
                            refine.refine_tracks(tracks, camera, Path('video.mp4'), options)
                    else:
                        result = refine.refine_tracks(tracks, camera, Path('video.mp4'), options)
                        self.assertTrue(result.meta['optimizer_fallback'])
                        self.assertEqual(result.meta['optimizer_stage'], 'filtered')
                client.close.assert_called_once()

    def test_fallback_exports_video_and_success_without_claiming_hmp_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            output.mkdir()
            video = root / 'source.mp4'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 30., (96, 64))
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(np.zeros((64, 96, 3), np.uint8))
            writer.release()
            tracks, camera = tracks_and_camera()
            options = deepcopy(DEFAULTS)
            for section in ('mask', 'tracking', 'jump', 'refine', 'depth_bounds', 'gaps'):
                options[section]['enabled'] = True
            options['refine'].update(workspace=str(root / 'optimizer'), camera_mode='author-static',
                                     mano=str(root / 'MANO_RIGHT.pkl'))
            client = Mock()
            client.request.return_value = {'ok': True, 'optimizer_fallback': True,
                                            'fallback_stage': 'hmp', 'fallback_reason': 'AssertionError: 128 frames'}
            with patch.object(refine, 'WorkerClient', return_value=client):
                result = refine.refine_tracks(tracks, camera, video, options['refine'])
            np.testing.assert_array_equal(result.arrays['keypoints_2d'], tracks.arrays['keypoints_2d'])
            final = derive(result, 'final')
            write_artifact(root / 'final.npz', final)
            publish_prediction(video, output, final, options)
            with patch('tools.run_wilor_video.run_pipeline'), contextlib.redirect_stdout(io.StringIO()):
                execute(video, output, options, 'full', None)
            success = json.loads((output / 'SUCCESS.json').read_text())
            self.assertTrue(success['optimizer_fallback'])
            self.assertFalse(success['prior_executed'])
            metadata = json.loads((output / HANDS_WILOR).with_suffix('.json').read_text())
            self.assertEqual(metadata['result_source'], 'wilor_filtered')
            with av.open(str(output / 'prediction.mp4')) as decoded:
                frames = list(decoded.decode(video=0))
            self.assertEqual(len(frames), 8)
            self.assertTrue(any(frame.to_ndarray().max() > 0 for frame in frames))

    def test_unrecorded_optimizer_failure_is_not_accepted_as_fallback(self):
        options = {section: {'enabled': True} for section in ('mask', 'tracking', 'jump', 'refine', 'depth_bounds', 'gaps')}
        artifact = SimpleNamespace(meta={'published_stage': 'final', 'pipeline_options': options,
                                         'optimizer_executed': False, 'boundary_trim': 3})
        with self.assertRaisesRegex(ValueError, '未实际完成'):
            validate_full_prediction(artifact)

    def test_missing_hmp_weights_do_not_block_the_prediction_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            bmc = repo / '_DATA/BMC'
            bmc.mkdir(parents=True)
            for i in range(7):
                np.save(bmc / f'{i}.npy', np.zeros(1))
            mano = repo / 'MANO_RIGHT.pkl'
            mano.write_bytes(b'model')
            options = deepcopy(DEFAULTS)
            options['wilor']['mano'] = str(mano)
            options['refine'].update(repo=str(repo), python=sys.executable,
                                     prior_config=str(repo / 'missing_prior_config.yaml'))
            with patch('egowhale.wilor.pipeline.runner.subprocess.check_output', return_value='revision'):
                identity = model_identity('refine', options)
            self.assertIn('prior_resources_issue', identity)


if __name__ == '__main__':
    unittest.main()
