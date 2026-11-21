"""Optional approach must preserve frame zero and downstream alignment."""
import json
from pathlib import Path

import numpy as np
import pytest

from egowhale.robots.motion import prefix_length


def test_prefix_missing_present_and_invalid(tmp_path):
    assert prefix_length(tmp_path) == 1
    (tmp_path / "action").mkdir()
    path = tmp_path / "action/prefix.npz"
    np.savez(path, joint_positions=np.zeros((5, 2)))
    assert prefix_length(tmp_path) == 5
    np.savez(path, joint_positions=np.zeros((0, 2)))
    with pytest.raises(ValueError, match="Empty approach"):
        prefix_length(tmp_path)


def test_render_frame_denominator_without_prefix(tmp_path):
    from egowhale.node_bench import processed_frames
    from egowhale.visual.render import Render
    assert processed_frames(Render(), tmp_path, 12) == (12, "output_video_frames")


@pytest.mark.parametrize("enabled", [True, False])
def test_dag_approach_dependency(tmp_path, enabled):
    from egowhale.batch.dag import Job
    job = Job(Path("/data/task/one.hdf5"), tmp_path, 4, spans=[(0, 2), (2, 4)], approach=enabled)
    assert (("approach", None) in job.units()) == enabled
    job.done.update({("retarget", None), ("base_ik", None)})
    assert (("render", 0) in job.ready()) == (not enabled)
    job.done.add(("approach", None))
    assert ("render", 0) in job.ready()


@pytest.mark.parametrize("with_prefix", [False, True])
@pytest.mark.parametrize("fps", [18.0, 30.0])
def test_composite_piece_alignment(tmp_path, monkeypatch, with_prefix, fps):
    from egowhale.pieces import piece_dir
    from egowhale.visual import composite
    spans = [(0, 2), (2, 4)]
    (tmp_path / "pieces.json").write_text(json.dumps(dict(frames=4, limit=2, pieces=spans)))
    if with_prefix:
        (tmp_path / "action").mkdir()
        np.savez(tmp_path / "action/prefix.npz", joint_positions=np.zeros((3, 2)))
    backgrounds, rendered = {}, {}
    for index, (start, stop) in enumerate(spans):
        where = piece_dir(tmp_path, index, len(spans))
        (where / "visual").mkdir(parents=True, exist_ok=True)
        background = np.broadcast_to(np.arange(start, stop, dtype=np.uint8)[:, None, None, None], (2, 2, 2, 3)).copy()
        backgrounds[where] = background
        n = 4 if with_prefix and index == 0 else 2
        rendered[where] = (np.zeros((n, 2, 2, 3), np.uint8), np.zeros((n, 2, 2), bool), np.zeros((n, 2, 2), bool), np.ones((n, 2, 2)))
        np.savez(where / composite.DEPTH, depth=np.ones((2, 2, 2)))
    monkeypatch.setattr(composite, "read_rgb", lambda p: (backgrounds[p.parent.parent], fps))
    monkeypatch.setattr(composite, "load_render", lambda p: rendered[p.parent.parent])
    monkeypatch.setattr(composite, "load_masks", lambda p: np.zeros((2, 2, 2), bool))
    outputs = []
    class Writer:
        def __init__(self, *args):
            assert args[1] == fps
        def write(self, array):
            outputs.append(array)
        def close(self):
            pass
    monkeypatch.setattr(composite, "RgbWriter", Writer)
    composite.Composite().run(Path("unused.hdf5"), tmp_path)
    expected = [0, 0, 0, 1, 2, 3] if with_prefix else [0, 1, 2, 3]
    assert np.concatenate(outputs)[:, 0, 0, 0].tolist() == expected


def test_node_bench_links_optional_prefix_only_when_present(tmp_path):
    from egowhale.node_bench import prepare
    from egowhale.visual.render import Render
    source = tmp_path / "inputs"
    (source / "action").mkdir(parents=True)
    (source / "pieces.json").write_text(json.dumps(dict(frames=2, limit=2, pieces=[[0, 2]])))
    for required in Render.needs:
        (source / required).write_bytes(b"upstream")
    for present in (False, True):
        if present:
            np.savez(source / "action/prefix.npz", joint_positions=np.zeros((3, 2)))
        dst = tmp_path / str(present)
        prepare(Path("unused.hdf5"), dst, source, Render(), 2)
        assert (dst / "action/prefix.npz").is_file() == present
        assert all((dst / name).is_file() for name in Render.needs)
