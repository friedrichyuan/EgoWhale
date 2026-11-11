"""CPU regressions for episode timebases and duration-preserving dataset subsampling."""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from egowhale.action import approach, base_ik, retarget
from egowhale.media import data_fps, read_chunks, read_rgb, valid_fps, video_shape, write_rgb


def fixture_episode(tmp_path, fps=30.0, count=10):
    src = tmp_path / "task" / "one.hdf5"
    src.parent.mkdir(parents=True)
    with h5py.File(src, "w") as f:
        f.create_dataset("camera/intrinsic", data=np.eye(3))
        f.create_dataset("transforms/camera", data=np.tile(np.eye(4), (count, 1, 1)))
        f.create_dataset("transforms/leftHand", data=np.tile(np.eye(4), (count, 1, 1)))
        f.create_dataset("confidences/leftHand", data=np.arange(count, dtype=np.float32))
    pixels = np.broadcast_to(np.arange(count, dtype=np.uint8)[:, None, None, None] * 20, (count, 16, 16, 3)).copy()
    write_rgb(src.with_suffix(".mp4"), pixels, fps)
    return src


@pytest.mark.parametrize("fps", [18.0, 24.0, 30.0])
def test_retarget_timebase_io_without_gpu(tmp_path, monkeypatch, fps):
    src = fixture_episode(tmp_path, fps)
    raw = retarget.read_episode(src)
    assert raw["fps"] == pytest.approx(fps)
    assert read_rgb(src.with_suffix(".mp4"))[1] == pytest.approx(fps)
    assert next(read_chunks(src.with_suffix(".mp4"), 0, 10))[0] == pytest.approx(fps)
    position = np.zeros((10, 2, 3))
    rotation = np.tile(np.eye(3), (10, 2, 1, 1))
    width, valid = np.zeros((10, 2)), np.ones((10, 2), bool)
    monkeypatch.setattr(retarget, "_episode_gpu", lambda data: (position, rotation, width, valid, np.eye(3)))
    monkeypatch.setattr(retarget, "_smooth_gpu", lambda p, r, w, v: (p, r, w))
    out = retarget.Retarget().compute(raw)
    retarget.save_gripper(tmp_path, out)
    assert base_ik.read_gripper(tmp_path)["fps"] == pytest.approx(fps)
    with h5py.File(src, "a") as f:
        f.attrs["egowhale_fps"] = fps + 1
    with pytest.raises(ValueError, match="mismatch"):
        retarget.read_episode(src)


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_invalid_fps_rejected(value):
    with pytest.raises(ValueError, match="FPS"):
        valid_fps(value)
    with pytest.raises(ValueError, match="FPS"):
        data_fps({"fps": value})


def test_legacy_default():
    assert data_fps({}) == 30


def test_velocity_uses_episode_timebase(monkeypatch):
    q = torch.arange(4, dtype=torch.float32)[:, None].repeat(1, 3)
    pos, quat = torch.zeros((4, 1, 3)), torch.zeros((4, 1, 4))
    quat[:, :, 0] = 1
    monkeypatch.setattr(base_ik, "_fk_camera", lambda *a: (pos, quat))
    args = (None, q, None, pos, quat, torch.ones((4, 1)), None, None, None, (torch.full((3,), -10.), torch.full((3,), 10.)))
    _, a = base_ik._loss(*args, fps=30)
    _, b = base_ik._loss(*args, fps=18)
    assert float(b["velocity"] / a["velocity"]) == pytest.approx((18 / 30) ** 2)
    for k in ("position", "orientation", "continuity", "jerk"):
        assert float(a[k]) == float(b[k])


def test_approach_resampling_duration():
    path = np.array([[0., 1.], [1., 2.], [2., 3.]])
    for fps in (18, 24, 30):
        sampled = approach._resample_path(path, 0.5, fps)
        assert len(sampled) == fps + 1
        np.testing.assert_array_equal(sampled[[0, -1]], path[[0, -1]])


def test_render_rejects_wrong_timebase_before_rendering(tmp_path, monkeypatch):
    from egowhale.visual import render
    src = fixture_episode(tmp_path, 18)
    (tmp_path / "action").mkdir()
    np.savez(tmp_path / "action/ik.npz", fps=30)
    monkeypatch.setattr(render, "_render", lambda *a: pytest.fail("must fail before GL/GPU work"))
    with pytest.raises(ValueError, match="IK/video FPS mismatch"):
        render.Render().run_piece(src, tmp_path, 0)


@pytest.mark.parametrize("fps", [24.0, 30.0])
@pytest.mark.parametrize("with_confidences", [True, False])
def test_converter_preserves_duration_and_alignment(tmp_path, fps, with_confidences):
    path = Path(__file__).resolve().parents[1] / "tools/downsample_egodex.py"
    spec = importlib.util.spec_from_file_location("downsample", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    src = fixture_episode(tmp_path, fps, 10)
    if not with_confidences:
        with h5py.File(src, "a") as f:
            del f["confidences"]
    out = tmp_path / "converted"
    result = module.convert(src, out, 18, 1)
    target = out / "test/task/one.hdf5"
    count, _, _, actual_fps = video_shape(target.with_suffix(".mp4"))
    assert count == 6 and actual_fps == pytest.approx(fps * .6)
    assert count / actual_fps == pytest.approx(10 / fps)
    with h5py.File(target) as f:
        assert f.attrs["egowhale_fps"] == pytest.approx(actual_fps)
        if with_confidences:
            np.testing.assert_array_equal(f["confidences/leftHand"][...], [0, 1, 3, 5, 6, 8])
        else:
            assert "confidences" not in f
        assert len(f["transforms/camera"]) == 6
    assert retarget.read_episode(target)["fps"] == pytest.approx(actual_fps)
    assert module.convert(src, out, 18, 1) == result
