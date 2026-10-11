"""Stage helpers that need no model: config validation, STTN windows, the Step base class, and retarget's GPU path."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from egowhale import config, paths
from egowhale.step import Result, Step, execute
from egowhale.visual.sttn import windows

EPISODE = paths.ROOT / "datasets" / "egodex" / "test" / "use_rubiks_cube" / "0.hdf5"


def _config(tmp_path, text: str) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text("episodes: x.txt\n" + text)
    return path


def test_config_fills_defaults_and_resolves_out(tmp_path):
    cfg = config.load(_config(tmp_path, "name: probe\n"))
    assert cfg["nodes"]["inpaint"] == {"cards": 1, "model": "sttn"}
    assert cfg["out"] == str(paths.ROOT / "outputs" / "probe")


@pytest.mark.parametrize(
    "text,message",
    [
        ("nodes:\n  inpaint: {model: lama}\n", "nodes.inpaint.model"),
        ("nodes:\n  action: {ride: segment}\n", "nodes.action.ride"),
        ("nodes:\n  action: {ride: depth, actors_per_card: 10, gpu_share: 0.1}\n", "whole card"),
        ("nodes:\n  inpaint: {cardz: 1}\n", "unknown config key nodes.inpaint.cardz"),
    ],
)
def test_config_errors_are_exceptions_not_exits(tmp_path, text, message):
    with pytest.raises(config.ConfigError, match=message):
        config.load(_config(tmp_path, text))


@pytest.mark.parametrize("radius,step,refs", [(5, 5, 12), (10, 20, 12), (10, 20, 6), (15, 30, 12)])
def test_sttn_windows_paint_every_frame_once_or_twice(radius, step, refs):
    for length in list(range(1, 80)) + [240, 471, 600]:
        made = windows(length, radius, step, refs)
        count = np.zeros(length, dtype=int)
        for neighbors, references in made:
            count[neighbors] += 1
            assert len(references) <= refs and not set(neighbors) & set(references)
            assert neighbors == list(range(neighbors[0], neighbors[-1] + 1))
        assert count.min() >= 1, (length, radius, step)


class _Touch(Step):
    name = "touch"
    makes = ("visual/touch.txt",)
    per_piece = True

    def __init__(self):
        super().__init__()
        self.ran = []

    def run_piece(self, src, dst, index):
        where, start, stop = self.piece(dst, index)
        (where / "visual").mkdir(parents=True, exist_ok=True)
        (where / "visual/touch.txt").write_text(f"{start}-{stop}")
        self.ran.append(index)


def test_per_piece_step_runs_only_missing_pieces(tmp_path):
    dst = tmp_path / "out"
    dst.mkdir()
    (dst / paths.PIECES).write_text(json.dumps({"frames": 9, "limit": 5, "pieces": [[0, 5], [5, 9]]}))
    step = _Touch()
    step.run_piece(tmp_path, dst, 1)
    assert step.todo(dst) == [0]
    step.run(tmp_path, dst)
    assert step.ran == [1, 0]
    assert (dst / "pieces/00/visual/touch.txt").read_text() == "0-5"
    assert execute(_Touch(), tmp_path, dst, 0) == Result(False)


def test_execute_reports_seconds_and_hides_prints(tmp_path, capsys):
    class Loud(_Touch):
        def run_piece(self, src, dst, index):
            print("noise")
            super().run_piece(src, dst, index)

    dst = tmp_path / "out"
    dst.mkdir()
    (dst / paths.PIECES).write_text(json.dumps({"frames": 3, "limit": 5, "pieces": [[0, 3]]}))
    result = execute(Loud(), tmp_path, dst, 0)
    assert result.ran and result.seconds >= 0 and result.error == ""
    assert "noise" not in capsys.readouterr().out


@pytest.mark.skipif(not EPISODE.is_file(), reason="EgoDex test data not present")
def test_retarget_gpu_matches_cpu_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from egowhale.action import retarget
    from tests import reference_retarget as reference

    raw = retarget.read_episode(EPISODE)
    position, rotation, width, valid, _intrinsic = retarget._episode_gpu(raw)
    cpu = reference.episode(EPISODE)
    assert (valid == cpu[3]).all()
    for got, want in zip((position, rotation, width), cpu[:3]):
        np.testing.assert_allclose(got[valid], want[valid], atol=1e-6)
    smoothed = retarget._smooth_gpu(position, rotation, width, valid)
    expected = reference.smooth(*cpu[:4])
    for got, want in zip(smoothed, expected):
        np.testing.assert_allclose(got[valid], want[valid], atol=1e-5)
