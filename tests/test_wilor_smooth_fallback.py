"""Exercise stage fallback and quality decisions without GPU inference."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from egowhale.wilor.synced import SYNC_ROOT

sys.path.insert(0, str(SYNC_ROOT))
from ego2robot.processors.hand_processor import HandProcessor
from ego2robot.utils.optimization_quality import assess_smooth_result

SPEC = importlib.util.spec_from_file_location('smooth_fallback_launcher', SYNC_ROOT / 'scripts/run_dynhamr.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def hand(frames=20, detected=True):
    xyz = np.zeros((frames, 21, 3), dtype=np.float32)
    xyz[..., 2] = .3
    return {'hand_detected': np.full(frames, detected), 'kpts_3d': xyz,
            'kpts_2d': np.tile([320., 180.], (frames, 21, 1)),
            'eef_rot': np.tile(np.eye(3), (frames, 1, 1))}


class QualityTests(unittest.TestCase):
    def assess(self, candidate, baseline=None):
        return assess_smooth_result(baseline or {'left': hand(), 'right': hand()}, candidate, 20, 640, 360, {})

    def test_small_adjustment_and_short_gap_loss_are_accepted(self):
        candidate = {'left': hand(), 'right': hand()}
        candidate['left']['hand_detected'][:3] = False
        candidate['right']['kpts_2d'] += 2
        quality = self.assess(candidate)
        self.assertTrue(quality['accepted'])
        self.assertAlmostEqual(quality['hands']['left']['retained_ratio'], .85)

    def test_empty_nonfinite_bad_projection_and_missing_side_are_rejected(self):
        for problem in ('empty', 'nan', 'negative_depth', 'displaced', 'missing', 'shape'):
            with self.subTest(problem=problem):
                candidate = {'left': hand(), 'right': hand()}
                if problem == 'empty':
                    candidate['left']['hand_detected'][:] = False
                elif problem == 'nan':
                    candidate['left']['kpts_3d'][1, 0, 0] = np.nan
                elif problem == 'negative_depth':
                    candidate['left']['kpts_3d'][1, 0, 2] = -1
                elif problem == 'displaced':
                    candidate['left']['kpts_2d'] += 200
                elif problem == 'missing':
                    del candidate['left']
                elif problem == 'shape':
                    candidate['left']['kpts_2d'] = np.zeros((20, 20, 2))
                self.assertFalse(self.assess(candidate)['accepted'])

    def test_single_hand_baseline_is_supported_but_no_observations_is_not(self):
        baseline = {'left': hand(detected=False), 'right': hand()}
        self.assertTrue(self.assess({'right': hand()}, baseline)['accepted'])
        baseline['right'] = hand(detected=False)
        self.assertFalse(self.assess({}, baseline)['accepted'])


class StageSelectionTests(unittest.TestCase):
    def test_static_camera_uses_supplied_intrinsics_with_identity_extrinsics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration = np.tile([343., 346., 426., 240.], (8, 1))
            np.savez(root / 'cameras.npz', width=852, height=480, intrins=calibration)
            class Camera:
                load_data = Mock()
            original = Camera.load_data
            launcher.install_static_camera_intrinsics(Camera)
            camera = Camera()
            camera.is_static = True
            camera.cam_dir = root
            camera.sidx, camera.eidx, camera.seq_len = 2, 7, 5
            camera.img_size = (426, 240)
            camera.load_data()
            np.testing.assert_allclose(camera.intrins.numpy(), calibration[2:7] / 2)
            np.testing.assert_allclose(camera.cam_R.numpy(), np.tile(np.eye(3), (5, 1, 1)))
            np.testing.assert_allclose(camera.cam_t.numpy(), 0)
            original.assert_not_called()
            camera.is_static = False
            camera.load_data()
            original.assert_called_once()

    def test_static_camera_rejects_missing_or_short_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            class Camera:
                load_data = Mock()
            launcher.install_static_camera_intrinsics(Camera)
            camera = Camera()
            camera.is_static = True
            camera.cam_dir = directory
            camera.sidx, camera.eidx, camera.seq_len = 0, 5, 5
            camera.img_size = (852, 480)
            with self.assertRaises(FileNotFoundError):
                camera.load_data()
            np.savez(Path(directory) / 'cameras.npz', width=852, height=480, intrins=np.ones((2, 4)))
            with self.assertRaises(ValueError):
                camera.load_data()

    def test_length_skip_or_hmp_failure_records_completed_smooth_path(self):
        for count in (94, 128, 266):
            with self.subTest(frames=count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                smooth = root / 'smooth_fit'
                smooth.mkdir()
                for step in (0, 300):
                    np.savez(smooth / f'episode_{step:06d}_world_results.npz', value=step)
                original = Mock(side_effect=RuntimeError('HMP weights unavailable'))
                fitting = SimpleNamespace(run_prior=original)
                status = root / 'hmp_fallback.json'
                launcher.install_hmp_fallback(fitting, status)
                fitting.run_prior(None, SimpleNamespace(seq_name='episode', seq_len=np.int64(count)), root)
                result = json.loads(status.read_text())
                self.assertEqual(result['skipped'], count != 128)
                self.assertTrue(result['root_smooth_completed'])
                self.assertEqual(result['smooth_result'], str((smooth / 'episode_000300_world_results.npz').resolve()))
                self.assertFalse((root / 'hmp_running.json').exists())
                self.assertEqual(original.call_count, int(count == 128))

    def test_successful_hmp_retains_existing_success_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fitting = SimpleNamespace(run_prior=Mock(return_value='prior-result'))
            launcher.install_hmp_fallback(fitting, root / 'hmp_fallback.json')
            self.assertEqual(fitting.run_prior(None, SimpleNamespace(seq_name='episode', seq_len=128), root), 'prior-result')
            self.assertFalse((root / 'hmp_fallback.json').exists())
            self.assertFalse((root / 'hmp_running.json').exists())
            self.assertTrue(json.loads((root / 'smooth_completed.json').read_text())['root_smooth_completed'])

    def test_split_checkpoint_accepts_dynhamr_prior_arguments(self):
        from omegaconf import OmegaConf

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            smooth = root / 'smooth_fit'
            smooth.mkdir()
            result = smooth / 'episode_000300_world_results.npz'
            np.savez(result, value=300)
            fitting = SimpleNamespace()
            launcher.checkpoint_prior(fitting, root)
            dataset = SimpleNamespace(seq_name='episode', seq_len=128, data_start=2, start_idx=3, track_ids=['001'])
            value = Mock()
            value.detach.return_value.cpu.return_value.numpy.return_value = np.ones((1, 128))
            observations = dict.fromkeys(('joints2d', 'vis_mask', 'is_right'), value)
            cfg = OmegaConf.create({'data': {'seq': 'episode'}})
            fitting.run_prior(cfg, dataset, root, 'cuda:0', ['smooth_fit'], observations,
                              None, cfg, cfg.data, root / 'prior')
            marker = json.loads((root / 'smooth_completed.json').read_text())
            self.assertEqual(marker, {'smooth_result': str(result), 'frames': 128,
                                      'frame_start': 5, 'tracks': ['001']})
            with np.load(root / 'observations.npz', allow_pickle=False) as saved:
                self.assertEqual(set(saved.files), set(observations))
                np.testing.assert_array_equal(saved['vis_mask'], np.ones((1, 128)))

    def test_smooth_is_published_only_after_acceptance(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                save = root / 'hand_processor'
                save.mkdir()
                for side in ('left', 'right'):
                    np.savez(save / f'hand_data_{side}.npz', **hand())
                originals = {p.name: p.read_bytes() for p in save.iterdir()}
                smooth = root / 'smooth_fit' / 'episode_000300_world_results.npz'
                smooth.parent.mkdir()
                smooth.touch()
                processor = object.__new__(HandProcessor)
                processor.cfg = {'dynhamr_root': str(root), 'dynhamr_required': True}
                processor._prepare_dynhamr_images = Mock()
                processor._convert_to_dynhamr = Mock()
                def failed_hmp(*args):
                    processor._hmp_fallback_reason = 'HMP requires 128 frames'
                    processor._smooth_result_path = str(smooth)
                    processor._hmp_skipped = True
                    return False
                processor._run_dynhamr_subprocess = Mock(side_effect=failed_hmp)
                candidate = {'left': hand(detected=valid), 'right': hand(detected=valid)}
                candidate['right']['kpts_2d'] += 2
                processor._postprocess_dynhamr = Mock(return_value=candidate)
                processor._run_dynhamr_refinement(str(root), save, 250, 250, 320, 180, 20, 360, 640, [])
                status = json.loads((root / 'optimization_status.json').read_text())
                self.assertEqual(status['result_source'], 'smooth_fit' if valid else 'wilor_filtered')
                self.assertEqual(status['succeeded'], valid)
                self.assertFalse(status['hmp_succeeded'])
                self.assertTrue(status['hmp_skipped'])
                self.assertEqual(status['quality']['accepted'], valid)
                processor._postprocess_dynhamr.assert_called_once()
                self.assertFalse(processor._postprocess_dynhamr.call_args.kwargs['write_output'])
                if valid:
                    with np.load(save / 'hand_data_right.npz') as result:
                        np.testing.assert_equal(result['kpts_2d'], candidate['right']['kpts_2d'])
                else:
                    self.assertEqual(originals, {p.name: p.read_bytes() for p in save.iterdir()})

    def test_missing_or_malformed_smooth_never_overwrites_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side in ('left', 'right'):
                np.savez(root / f'hand_data_{side}.npz', **hand())
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            processor = object.__new__(HandProcessor)
            processor.cfg = {}
            args = (root, root, 'episode', root, 250, 250, 320, 180, 20, 360, 640, 3, .15, .4)
            self.assertFalse(processor._try_smooth_fallback(*args)['accepted'])
            smooth = root / 'smooth_fit' / 'episode_000300_world_results.npz'
            smooth.parent.mkdir()
            smooth.touch()
            processor._smooth_result_path = smooth
            processor._postprocess_dynhamr = Mock(side_effect=ValueError('corrupt MANO arrays'))
            self.assertFalse(processor._try_smooth_fallback(*args)['accepted'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.glob('*.npz')})

    def test_hmp_worker_crash_and_timeout_keep_smooth_reference(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / 'smooth_fit' / 'episode_000300_world_results.npz'
                source.parent.mkdir()
                source.touch()
                processor = object.__new__(HandProcessor)
                processor.cfg = {'dynhamr_log_dir': str(root)}
                def worker(*args, **kwargs):
                    (root / 'hmp_running.json').write_text(json.dumps({'smooth_result': str(source)}))
                    if timeout:
                        raise subprocess.TimeoutExpired('HMP', 1)
                    return subprocess.CompletedProcess([], -9, '', 'worker killed')
                with patch('ego2robot.processors.hand_processor.run_streamed_command', side_effect=worker):
                    self.assertFalse(processor._run_dynhamr_subprocess(str(root), str(root), 'episode'))
                self.assertEqual(processor._smooth_result_path, str(source))
                self.assertIn('HMP worker', processor._hmp_fallback_reason)

    def test_root_smooth_failure_still_honors_required_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            processor = object.__new__(HandProcessor)
            processor.cfg = {'dynhamr_root': directory, 'dynhamr_required': True}
            processor._prepare_dynhamr_images = Mock()
            processor._convert_to_dynhamr = Mock()
            processor._run_dynhamr_subprocess = Mock(return_value=False)
            with self.assertRaisesRegex(RuntimeError, 'DynHaMR optimization failed'):
                processor._run_dynhamr_refinement(directory, Path(directory), 250, 250, 320, 180, 20, 360, 640, [])


if __name__ == '__main__':
    unittest.main()
