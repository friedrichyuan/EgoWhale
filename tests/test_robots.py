"""CPU contract tests for downloaded assets and standalone node boundaries."""
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from egowhale.action import approach, base_ik
from egowhale.robots import load_robot
from egowhale.robots.motion import read_bases, read_motion


@pytest.mark.parametrize("name,count", [("arx", 1), ("piper", 2)])
def test_assets_and_named_motion(name, count):
    robot = load_robot(name)
    assert len(robot.groups) == count
    model = robot.model
    values = np.zeros((5, model.nq), dtype=np.float32)
    for group in robot.groups:
        for i, joint in enumerate(group.joints):
            values[:, model.joint(joint).qposadr[0]] = group.home[i]
    named = robot.named(values)
    order = np.arange(len(named["joint_names"]))[::-1]
    shuffled = dict(joint_names=named["joint_names"][order], joint_positions=named["joint_positions"][:, order])
    np.testing.assert_array_equal(robot.qpos(shuffled), values)
    assert all(model.site(site).id >= 0 for site in robot.tcp_sites)
    assert sum(robot.is_gripper(model, i) for i in range(model.ngeom)) > 0
    bases = {g.name: np.eye(4) for g in robot.groups}
    for i, matrix in enumerate(bases.values()):
        matrix[0, 3] = .25 * (i + 1)
    robot.place_bases(model, bases)
    for g in robot.groups:
        np.testing.assert_allclose(model.body(robot.roots[g.name]).pos, bases[g.name][:3, 3])


def test_arx_matches_legacy_pack_and_seed():
    from pathlib import Path
    golden = np.load(Path(__file__).parent / "fixtures/arx_adapter_golden.npz")
    robot = load_robot("arx")
    rng = np.random.default_rng(123)
    arms = rng.uniform(-.1, .1, (10, 12))
    width = rng.uniform(0., .12, (10, 2))
    valid = rng.random((10, 2)) > .2
    np.testing.assert_array_equal(robot.qpos(robot.pack([arms], width, valid)), golden["qpos"])
    position = rng.normal(size=(10, 2, 3))
    a = robot.groups[0].seed(position, valid)
    b = (golden["rotation"], golden["translation"])
    for got, want in zip(a, b):
        np.testing.assert_allclose(got, want, atol=1e-15)
    np.testing.assert_array_equal(base_ik._offset_grid(robot.groups[0].bounds), golden["grid"])


def test_piper_urdf_matches_mjcf_fk():
    robot = load_robot("piper")
    model = robot.model
    data = mujoco.MjData(model)
    rng = np.random.default_rng(12)
    for group in robot.groups:
        tree = ET.fromstring(robot._urdf(group.name))
        for _ in range(12):
            q = rng.uniform(group.lower, group.upper)
            data.qpos[:] = 0.
            for name, value in zip(group.joints, q):
                data.qpos[model.joint(name).qposadr[0]] = value
            mujoco.mj_forward(model, data)
            poses = {robot.roots[group.name]: np.eye(4)}
            for joint in tree.findall("joint"):
                origin = joint.find("origin")
                transform = np.eye(4)
                transform[:3, :3] = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy"), sep=" ")).as_matrix()
                transform[:3, 3] = np.fromstring(origin.get("xyz"), sep=" ")
                if joint.get("type") == "revolute":
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                    transform[:3, :3] = transform[:3, :3] @ Rotation.from_rotvec(axis * q[list(group.joints).index(joint.get("name"))]).as_matrix()
                poses[joint.find("child").get("link")] = poses[joint.find("parent").get("link")] @ transform
            got = poses[group.tools[0]]
            np.testing.assert_allclose(got[:3, 3], data.site(group.tools[0]).xpos, atol=1e-9)
            np.testing.assert_allclose(got[:3, :3], data.site(group.tools[0]).xmat.reshape(3, 3), atol=1e-9)
    fingers = robot.gripper_positions(np.full((2, 2), .04), np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(fingers["left_joint7"], -fingers["left_joint8"])


@pytest.mark.parametrize("name", ["arx", "piper"])
def test_node_io_and_group_dispatch_without_gpu(tmp_path, monkeypatch, name):
    calls = []
    def solve(position, rotation, valid, group):
        calls.append((group.name, position.copy()))
        return np.eye(4), np.repeat(group.home[None], len(position), axis=0), {"position": 0., "jerk": 0.}
    monkeypatch.setattr(base_ik, "_solve", solve)
    data = dict(position=np.zeros((8, 2, 3)), rotation=np.tile(np.eye(3), (8, 2, 1, 1)),
                width=np.full((8, 2), .04), valid=np.ones((8, 2), dtype=bool))
    data["position"][:, 1] = 1.
    node = base_ik.BaseIK(name)
    result = node.compute(data)
    node.save(tmp_path, result)
    groups = node.robot.groups
    assert len(calls) == len(groups)
    for (_, received), group in zip(calls, groups):
        np.testing.assert_array_equal(received, data["position"][:, list(group.sides)])
    loaded = read_motion(tmp_path / "action/ik.npz")
    assert len(read_bases(tmp_path / "action/base.json")) == len(groups)
    def trajectory(goal, group):
        count = 10 if group.sides[0] == 0 else 16
        return np.linspace(group.home, goal, count)
    monkeypatch.setattr(approach, "_trajopt", trajectory)
    # A new node instance uses only persisted motion and its own adapter.
    other = approach.Approach(name)
    prefix = other.compute(other.read(None, tmp_path))
    other.save(tmp_path, prefix)
    np.testing.assert_allclose(prefix["joint_positions"][-1], loaded["joint_positions"][0])
    assert len(prefix["joint_positions"]) == (10 if name == "arx" else 16)


@pytest.mark.parametrize("name", ["arx", "piper"])
@pytest.mark.parametrize("with_prefix", [True, False])
@pytest.mark.parametrize("split", [True, False])
def test_render_and_actor_boundaries(tmp_path, monkeypatch, name, with_prefix, split):
    import json

    from egowhale.batch.actors import ActionActor, ActionReadActor, ActionWriteActor, CpuActor, RenderActor
    from egowhale.visual import render
    robot = load_robot(name)
    (tmp_path / "action").mkdir()
    spans = [[0, 2], [2, 4]] if split else [[0, 4]]
    (tmp_path / "pieces.json").write_text(json.dumps({"frames": 4, "limit": 2 if split else 10, "pieces": spans}))
    motion = robot.pack([np.repeat(g.home[None], 4, axis=0) for g in robot.groups],
                        np.full((4, 2), .04), np.ones((4, 2), dtype=bool))
    motion["joint_positions"][:, 0] += np.arange(4) * .01
    np.savez_compressed(tmp_path / "action/ik.npz", **motion)
    if with_prefix:
        np.savez_compressed(tmp_path / "action/prefix.npz", **motion)
    np.savez_compressed(tmp_path / "action/gripper.npz", intrinsic=np.array([[50., 0., 8.], [0., 50., 8.], [0., 0., 1.]]))
    (tmp_path / "action/base.json").write_text(json.dumps({"bases": {g.name: np.eye(4).tolist() for g in robot.groups}}))
    monkeypatch.setattr(render, "frame_size", lambda _: (16, 16))
    seen = []
    seen_rows = []
    def raster(qpos, bases, intrinsic, h, w, selected):
        # Exercise actual asset and camera setup, without allocating an OpenGL context.
        model = selected.scene_spec().compile()
        selected.place_bases(model, bases)
        selected.place_camera(model, intrinsic, h)
        selected.hide(model)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos[0]
        mujoco.mj_forward(model, data)
        seen.append((selected.name, len(qpos)))
        seen_rows.append(qpos.copy())
        return (np.zeros((len(qpos), h, w, 3), dtype=np.uint8), np.ones((len(qpos), h, w), bool),
                np.zeros((len(qpos), h, w), bool), np.ones((len(qpos), h, w), np.float32))
    monkeypatch.setattr(render, "_render", raster)
    from egowhale.pieces import piece_dir
    for index in range(len(spans)):
        render.Render(name).run_piece(tmp_path / "source.hdf5", tmp_path, index)
        assert (piece_dir(tmp_path, index, len(spans)) / "visual/render.npz").is_file()
    qpos = robot.qpos(motion)
    expected = np.concatenate((qpos, qpos[1:])) if with_prefix else qpos
    np.testing.assert_array_equal(np.concatenate(seen_rows), expected)
    assert sum(count for _, count in seen) == len(expected)
    # Use the same constructors invoked remotely by make_actors, without launching Ray/GPU models.
    for cls, stage in [(ActionActor, "base_ik"), (ActionReadActor, "approach"),
                       (ActionWriteActor, "base_ik"), (RenderActor, "render"), (CpuActor, "curate")]:
        actor = cls(name)
        assert actor.steps[stage].robot.name == robot.name
