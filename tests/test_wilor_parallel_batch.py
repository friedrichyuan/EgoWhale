"""Parallel WiLoR contracts with deterministic workers and no model inference."""
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from egowhale.wilor.parallel_stages import execute, read_candidates, save_candidates
from egowhale.wilor.retarget_export import BATCH_RETARGET_HDF5, export_existing
from egowhale.wilor.synced import CONFIG, RUNNER, identity
from egowhale.wilor.pipeline.hmp import windows

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from wilor_parallel_batch import Batch, Job


class FakeClient:
    def __init__(self, *_):
        pass

    def request(self, *_):
        return 0

    def close(self):
        pass


def candidate(score):
    return SimpleNamespace(bbox=[0, 0, 10, 10], score=score, is_right=True,
                           keypoints_3d=np.zeros((21, 3)), cam_t=[0, 0, 1],
                           focal_length=2500., mano_pose=np.zeros(48), mano_shape=np.zeros(10))


class WiLoRParallelTests(unittest.TestCase):
    def test_candidate_blocks_keep_empty_frames_and_official_tie_order(self):
        predictions = [[candidate(.5), candidate(.5), candidate(.4)], [], [candidate(.6)]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for keep_all, expected in [(False, [1, 0, 1]), (True, [3, 0, 1])]:
                path = root / str(keep_all)
                save_candidates(iter(predictions), path, 3, keep_all, block_size=2)
                result = list(read_candidates(path, 3))
                self.assertEqual(list(map(len, result)), expected)
                self.assertEqual(result[0][0].score, .5)
                np.testing.assert_array_equal(result[2][0].cam_t, [0, 0, 1])

    def test_basic_calibration_and_publish_keep_the_source_timeline(self):
        ffmpeg_path = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
        if not shutil.which("ffmpeg", path=ffmpeg_path):
            self.skipTest("ffmpeg unavailable")
        spec = importlib.util.spec_from_file_location("test_parallel_experiment", RUNNER)
        experiment = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(experiment)
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"PATH": ffmpeg_path}):
            root = Path(directory)
            source = root / "input.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 18., (80, 48))
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(np.zeros((48, 80, 3), dtype=np.uint8))
            writer.release()
            dst = root / "job"
            args = {"video": str(source), "output": str(dst), "config": str(CONFIG),
                    "identity": identity(source), "signature": "cpu-smoke"}

            def run(stage, **extra):
                execute({**args, **extra, "stage": stage, "attempt_dir": str(root / stage)}, experiment, None)

            run("prepare")
            work = dst / "action/wilor_ego2robot"
            (work / "cam_intrinsics.json").write_text(json.dumps(
                {"left": {"fx": 50, "fy": 50, "cx": 40, "cy": 24}}))
            row = candidate(.8)
            row.focal_length = 100.
            row.cam_t = [0, 0, .6]
            save_candidates(([row] for _ in range(8)), root / "candidates", 8, False)
            run("basic", candidates=str(root / "candidates"))
            run("publish", retarget_hdf5=BATCH_RETARGET_HDF5)
            with np.load(dst / "action/hands_wilor.npz") as data:
                np.testing.assert_array_equal(data["frame_index"], np.arange(8))
                np.testing.assert_allclose(data["timestamps"], np.arange(8) / 18)
                self.assertTrue(data["valid"][:, 1].all())
                self.assertFalse(data["valid"][:, 0].any())
                np.testing.assert_allclose(data["joints_camera"][:, 1, :, 2], .3)
                from egowhale.action.retarget import read_episode

                hdf5 = dst / BATCH_RETARGET_HDF5
                exported = read_episode(hdf5)
                np.testing.assert_array_equal(exported["intrinsic"], data["intrinsic"])
                np.testing.assert_array_equal(exported["camera"], np.broadcast_to(np.eye(4), (8, 4, 4)))
                for joint, index in (("Hand", 0), ("ThumbTip", 4), ("IndexFingerTip", 8), ("MiddleFingerTip", 12)):
                    np.testing.assert_array_equal(exported["joints"]["right" + joint][:, :3, 3],
                                                  data["joints_camera"][:, 1, index])
                    self.assertTrue(np.isnan(exported["joints"]["left" + joint][:, :3, 3]).all())
                self.assertEqual(hdf5.with_suffix(".mp4").resolve(), source)
            self.assertEqual(json.loads((work / "annotation_counts.json").read_text())["right"], 8)
            self.assertEqual(export_existing(dst), hdf5)
            (root / "final_videos").mkdir()
            job = Job(1, source, dst, args["identity"], args["signature"], done={"publish": root / "publish"})
            batch = Batch({}, {"cards": {}, "required": False}, CONFIG, Path(sys.executable), root, [job], False, {})
            writer = Mock()
            batch.record(job, writer)
            row = writer.writerow.call_args.args[0]
            self.assertEqual(Path(row["retarget_hdf5"]), hdf5)
            self.assertEqual(Path(row["video"]).parent, root / "final_videos")
            self.assertEqual(Path(row["video"]).resolve(), work / "optimized_hand_points.mp4")
            self.assertFalse((root / "retarget_inputs").exists())
            self.assertEqual(list(root.rglob("*.hdf5")), [hdf5])

    def test_hmp_window_ranges(self):
        self.assertEqual(windows(127), [(0, 127)])
        self.assertEqual(windows(128), [(0, 128)])
        self.assertEqual(windows(129), [(0, 128), (112, 129)])
        self.assertEqual(windows(266), [(0, 128), (112, 240), (224, 266)])

    def test_gpu_branches_overlap_and_failed_window_falls_back(self):
        with tempfile.TemporaryDirectory() as directory, patch("wilor_parallel_batch.ResidentClient", FakeClient):
            root = Path(directory)
            (root / "final_videos").mkdir()
            job = Job(1, root / "test.mp4", root / "one", {}, "mock-signature")
            cards = dict(camera=[0], wilor=[1], mask=[2], optimizer=[3, 4])
            config = root / "config.yaml"
            config.write_text("{}")
            batch = Batch({}, {"cards": cards, "required": False}, config, Path(sys.executable),
                          root, [job], True, {})
            ready_together = threading.Barrier(3)
            windows_seen = []
            merge_requests = []
            record = []

            def work(slot, task):
                stage = task.stage
                attempt = root / (task.key + f"_{task.trial}")
                attempt.mkdir()
                if stage in ("camera", "candidates", "masks"):
                    ready_together.wait(timeout=5)
                if stage == "smooth":
                    output = attempt / "smooth"
                    output.mkdir()
                    (output / "smooth_completed.json").write_text(json.dumps({"frames": 266, "tracks": ["001"]}))
                if stage == "hmp":
                    windows_seen.append((slot, task.args["start"], task.trial))
                    if task.args["start"] == 224:
                        return 1, attempt, 1.0, "failed"
                if stage == "merge":
                    merge_requests.append(task.args)
                return 0, attempt, 1.0, "completed"

            def terminal(job, _writer):
                record.append(job)
                batch.rows.append({"status": "成功", "frames": 266, "result_source": "smooth_fit"})

            with patch.object(batch, "work", side_effect=work), patch.object(batch, "record", side_effect=terminal):
                self.assertEqual(batch.run(), 0)
            self.assertEqual(len(record), 1)
            self.assertIn("publish", record[0].done)
            self.assertEqual(len(merge_requests), 1)
            self.assertIn("hmp", merge_requests[0]["fallback_reason"])
            self.assertEqual({start for _, start, _ in windows_seen}, {0, 112, 224})
            self.assertEqual(sum(start == 224 for _, start, _ in windows_seen), 2)


if __name__ == "__main__":
    unittest.main()
