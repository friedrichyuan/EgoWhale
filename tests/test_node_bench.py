import json
from pathlib import Path

import numpy as np
import pytest

from egowhale.node_bench import benchmark, processed_frames, summarize
from egowhale.step import Step


class FakeAction(Step):
    name = "base_ik"
    needs = ("action/gripper.npz",)
    makes = ("action/ik.npz", "action/base.json")

    def __init__(self):
        super().__init__()
        self.loads = self.releases = 0

    def load(self):
        self.loads += 1

    def release(self):
        self.releases += 1

    def read(self, src, dst):
        return len(np.load(dst / self.needs[0])["position"])

    def compute(self, data):
        return np.zeros((data, 3))

    def save(self, dst, result):
        np.savez_compressed(dst / self.makes[0], joint_positions=result)
        (dst / self.makes[1]).write_text("{}")


def setup_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    for name in ("one", "two"):
        folder = inputs / "task" / name
        (folder / "action").mkdir(parents=True)
        (folder / "pieces.json").write_text(json.dumps(dict(limit=4, frames=6, pieces=[[0, 3], [3, 6]])))
        np.savez_compressed(folder / "action/gripper.npz", position=np.zeros((6, 2, 3)))
        (folder / "action/ik.npz").write_bytes(b"old output must not be reused")
    return inputs, [Path("/data/task/one.hdf5"), Path("/data/task/two.hdf5")]


def test_batch_reuses_node_links_inputs_excludes_warmup(tmp_path):
    inputs, sources = setup_inputs(tmp_path)
    node = FakeAction()
    result = benchmark(node, sources, tmp_path / "out", inputs, warmup=1)
    assert node.loads == node.releases == 1
    assert result["ok"] == 2 and result["failed"] == 0
    assert result["all_successful"]["frames"] == 12
    assert result["after_warmup"]["frames"] == 6
    assert result["after_warmup"]["compute_fps"] > 0
    assert (tmp_path / "out/task/one/action/gripper.npz").is_symlink()
    assert not (tmp_path / "out/task/one/action/ik.npz").is_symlink()
    assert (inputs / "task/one/action/ik.npz").read_bytes() == b"old output must not be reused"
    assert json.loads((tmp_path / "out/task/one/pieces.json").read_text())["limit"] == 4
    with pytest.raises(FileExistsError):
        benchmark(node, sources, tmp_path / "out", inputs)


def test_missing_input_excluded_and_next_episode_runs(tmp_path):
    inputs, sources = setup_inputs(tmp_path)
    (inputs / "task/one/action/gripper.npz").unlink()
    result = benchmark(FakeAction(), sources, tmp_path / "out", inputs, warmup=0)
    assert result["failed"] == result["ok"] == 1
    assert result["all_successful"]["frames"] == 6
    rows = [json.loads(line) for line in (tmp_path / "out/results.jsonl").read_text().splitlines()]
    assert "frames" not in rows[0]
    assert rows[1]["status"] == "ok"


def test_weighted_aggregate_and_zero_measurements():
    rows = [dict(status="ok", frames=10, node_s=1, warmup=True),
            dict(status="ok", frames=40, node_s=4, warmup=False),dict(status="failed")]
    s = summarize(rows, 2, 10)
    assert s["overall_fps"] == 5
    assert s["all_successful"]["node_fps"] == 10
    assert s["after_warmup"]["node_fps"] == 10
    assert s["after_warmup"]["compute_fps"] is None
    assert summarize([], 0, 1)["all_successful"]["node_fps"] is None


def test_render_counts_prefix_and_only_processed_frames(tmp_path):
    (tmp_path / "action").mkdir()
    np.savez_compressed(tmp_path / "action/prefix.npz", joint_positions=np.zeros((5, 4)))
    node = FakeAction()
    node.name = "render"
    assert processed_frames(node, tmp_path, 12) == (16, "output_video_frames")
