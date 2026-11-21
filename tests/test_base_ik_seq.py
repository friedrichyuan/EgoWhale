import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from egowhale.action import base_ik_seq as module
from egowhale.batch.dag import Job
from egowhale.batch.episode_actions import EpisodeActions, choose


class Backend:
    def __init__(self):
        self.trace = []

    def prepare(self, data):
        if data < 0:
            raise ValueError("bad input")
        return dict(frame=0, length=data, history=[])

    def step(self, states):
        for s in states:
            s["frame"] += 1
            s["history"].append(s["frame"])
        self.trace.append([s["frame"] for s in states])

    def finish(self, state):
        return state["history"]

    def reset(self):
        pass


def stream(backend=None):
    return module.EpisodeStream("piper", 324, backend or Backend())


def job(i):
    j = Job(Path(f"task/{i}.hdf5"), Path(f"/missing/stream-test/{i}"), 2, spans=[(0, 2)], approach=False)
    j.done = {("retarget", None)}
    return j


def test_capacity_and_config(tmp_path):
    from egowhale.config import ConfigError, load

    assert module.episode_capacity("piper", 2160) == 8
    assert module.episode_capacity("arx", 2160) == 16
    with pytest.raises(ValueError):
        module.episode_capacity("piper", 200)
    p = tmp_path / "c.yaml"
    p.write_text("robot: piper\nepisodes: a.hdf5\nnodes: {action: {batch_size: 2160}}\nmemory: {inflight: 7}\n")
    with pytest.raises(ConfigError, match="inflight"):
        load(p)


def test_finish_refill_tail_and_state_isolation():
    s = stream()
    rows = s.advance([("long", 20), ("short", 4)])
    assert [(r["token"], r["out"]) for r in rows] == [("short", [1, 2, 3, 4])]
    assert s.active["long"]["frame"] == 4
    rows = s.advance([("replacement", 5)])
    assert rows[0]["token"] == "replacement" and rows[0]["out"] == [1, 2, 3, 4, 5]
    assert s.active["long"]["frame"] == 9
    all_rows = []
    while s.active:
        all_rows.extend(s.advance())
    assert all_rows[0]["out"] == list(range(1, 21))
    assert s.advance() == []


def test_admission_and_solver_failures():
    s = stream()
    rows = s.advance([("bad", -1), ("good", 20)], steps=1)
    assert rows[0]["ok"] is False and set(s.active) == {"good"}
    with pytest.raises(ValueError, match="duplicate"):
        s.advance([("good", 4)])
    with pytest.raises(ValueError, match="capacity"):
        s.advance([("a", 4), ("b", 4)])

    def fail(_):
        raise RuntimeError("solver failure")

    s.backend.step = fail
    rows = s.advance()
    assert rows[0]["token"] == "good" and not rows[0]["ok"] and not s.active


def test_individual_finalize_failure():
    b = Backend()

    def finish(s):
        if s["length"] == 4:
            raise ValueError("bad finalization")
        return s["history"]

    b.finish = finish
    s = stream(b)
    rows = s.advance([("bad", 4), ("good", 8)])
    assert not rows[0]["ok"] and set(s.active) == {"good"}
    assert s.advance()[0]["out"] == list(range(1, 9))


def test_iterator_yields_before_loading_replacement(monkeypatch):
    original = module.EpisodeStream
    monkeypatch.setattr(module, "EpisodeStream", lambda r, b: original("piper", 324, Backend()))
    reads = []

    def source():
        for key, length in [(0, 20), (1, 4), (2, 5)]:
            reads.append(key)
            yield key, length

    results = module.iter_results(None, source(), 324)
    assert next(results)["token"] == 1
    assert reads == [0, 1]
    assert next(results)["token"] == 2
    assert reads == [0, 1, 2]
    assert next(results)["token"] == 0
    with pytest.raises(StopIteration):
        next(results)
    assert module.compute_batch(None, [20, 4, 5], 324) == [list(range(1, n + 1)) for n in [20, 4, 5]]


def test_refill_does_not_wait_for_full_batch():
    active = [job(0)]
    producer = job(1)
    producer.done.clear()
    producer.running.add(("retarget", None))
    active.append(producer)
    assert choose(active, 2) is None
    assert choose(active, 2, refill=True) == ("base_ik", active[:1])
    producer.failed = True
    assert choose(active, 2) == ("base_ik", active[:1])


def test_scheduler_early_write_replacement_and_drain(monkeypatch):
    values = {}
    active = [job(i) for i in range(3)]
    lengths = {str(j.dst): n for j, n in zip(active, [20, 4, 5])}
    engine = stream()
    written = []

    def remote(fn):
        def submit(*args):
            ref = object()
            values[ref] = (fn, args)
            return ref

        return SimpleNamespace(remote=submit)

    def get(ref):
        value = values[ref]
        if isinstance(value, tuple):
            fn, args = value
            value = fn(*args)
            values[ref] = value
        return value

    def load(stage, units):
        return [dict(ok=True, dst=dst, data=lengths[dst]) for src, dst in units]

    def advance(payload, batch_size, start, count):
        data = [] if payload is None else get(payload)[start : start + count]
        rows = engine.advance([(p["dst"], p["data"]) for p in data])
        for row in rows:
            row["seconds"] = 1.0
            row["save_ref"] = row["token"]
        return dict(rows=rows, live=list(engine.active))

    def save(token):
        written.append((token, list(engine.active)))

    lane = SimpleNamespace(
        reader=SimpleNamespace(load_many=remote(load)),
        actor=SimpleNamespace(advance_ik=remote(advance)),
        writer=SimpleNamespace(save=remote(save)),
    )
    pending = {}

    def apply(units, rows, stage):
        for j, _ in units:
            j.running.discard((stage, None))
            j.done.add((stage, None))

    def on_write(ref, stage, units, rows):
        get(ref)
        apply(units, rows, stage)

    scheduler = SimpleNamespace(action_lanes=[lane], active=active, pending=pending, apply=apply, on_write=on_write)
    manager = EpisodeActions(scheduler, 2, 324)
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(get=get))
    for _ in range(100):
        manager.feed()
        if not pending:
            break
        ref = next(iter(pending))
        handler, args = pending.pop(ref)
        handler(ref, *args)
    else:
        pytest.fail("scheduler did not drain")
    assert [t for t, _ in written] == [str(active[i].dst) for i in [1, 2, 0]]
    assert str(active[0].dst) in written[0][1]
    assert all(("base_ik", None) in j.done and not j.running for j in active)


@pytest.mark.parametrize(("name", "capacity"), [("piper", 8), ("arx", 2)])
def test_real_backend_fixed_width_and_episode_state(monkeypatch, name, capacity):
    from egowhale.action import base_ik as core
    from egowhale.action.sequential_ik import CuroboBackend
    from egowhale.robots import load_robot

    robot = load_robot(name)
    groups = {id(g.config): g for g in robot.groups}
    allocated_widths = []

    class Solver:
        def __init__(self, cfg):
            self.group = groups[id(cfg["robot"])]
            self.joint_names = list(self.group.joints)[::-1]
            self.width = cfg["max_batch_size"]
            allocated_widths.append(self.width)
            self.seed_ik_solver = SimpleNamespace(config=SimpleNamespace(use_cuda_graph=True))

        def solve_pose(self, goal_tool_poses, current_state, return_seeds):
            # Model a cuRobo solver that reuses buffers sized for the configured batch.
            assert len(current_state.position) == self.width
            assert not self.seed_ik_solver.config.use_cuda_graph
            for pose in goal_tool_poses.values():
                assert len(pose.position) == len(pose.quaternion) == self.width
                torch.testing.assert_close(pose.quaternion.norm(dim=-1), torch.ones(self.width))
            values = current_state.position + 0.001
            home = torch.as_tensor(self.group.home.copy()[::-1].copy(), dtype=values.dtype)
            first = (current_state.position == home).all(dim=-1, keepdim=True)
            distant = torch.where(first, values, values + 0.5)
            # Exercise returned joint order differing from solver order.
            return SimpleNamespace(
                js_solution=SimpleNamespace(
                    position=torch.stack((distant, values), dim=1).flip(-1),
                    joint_names=list(self.group.joints),
                ),
                success=torch.ones((len(values), return_seeds), dtype=torch.bool),
                position_error=torch.zeros((len(values), return_seeds)),
                rotation_error=torch.zeros((len(values), 1)),
            )

    types = SimpleNamespace(
        GoalToolPose=SimpleNamespace(from_poses=lambda poses, **kw: poses),
        JointState=SimpleNamespace(
            from_position=lambda p, joint_names: SimpleNamespace(position=p, joint_names=joint_names)
        ),
        Pose=lambda **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setitem(sys.modules, "curobo.types", types)
    monkeypatch.setitem(
        sys.modules,
        "curobo.inverse_kinematics",
        SimpleNamespace(InverseKinematics=Solver, InverseKinematicsCfg=SimpleNamespace(create=lambda **kw: kw)),
    )
    monkeypatch.setitem(
        sys.modules,
        "curobo.kinematics",
        SimpleNamespace(
            Kinematics=lambda cfg: None, KinematicsCfg=SimpleNamespace(from_robot_yaml_file=lambda *a, **kw: None)
        ),
    )
    original = torch.as_tensor
    monkeypatch.setattr(
        torch, "as_tensor", lambda x, **kw: original(x, **{k: v for k, v in kw.items() if k != "device"})
    )
    seen_fps = []
    def refine(kin, p, r, v, sr, st, o, a, g, fps, geometry=None, periodic=None):
        seen_fps.append(fps)
        return o, a, {"position": 0.0}
    monkeypatch.setattr(core, "_refine", refine)

    def data(n, fps=30.0):
        return dict(
            fps=fps,
            position=np.zeros((n, 2, 3)),
            rotation=np.tile(np.eye(3), (n, 2, 1, 1)),
            valid=np.ones((n, 2), bool),
            width=np.full((n, 2), 0.04),
        )

    batch_size = capacity * sum(len(core._offset_grid(g.bounds)) for g in robot.groups)
    s = module.EpisodeStream(robot, batch_size, CuroboBackend(robot, capacity))
    extra = [(i, data(11)) for i in range(3, capacity + 1)]
    first = s.advance([(0, data(20)), (1, data(4, 18.0)), *extra])[0]
    assert first["token"] == 1 and first["out"]["fps"] == 18.0
    assert seen_fps == [18.0] * len(robot.groups)
    # At this point the PiPer batch has 1134 real rows after starting with 1296.
    assert len(s.active) == capacity - 1
    assert s.advance(steps=1) == []
    row = s.advance([(2, data(5))])[0]
    assert row["token"] == 2 and row["out"]["fps"] == 30.0
    assert seen_fps == [18.0] * len(robot.groups) + [30.0] * len(robot.groups)
    q = row["out"]["joint_positions"]
    names = list(row["out"]["joint_names"])
    for g in robot.groups:
        np.testing.assert_allclose(
            q[:, [names.index(j) for j in g.joints]], g.home + np.arange(1, 6)[:, None] * 0.001, atol=1e-6
        )
    assert s.active[0]["frame"] == 10
    assert [r["token"] for r in s.advance(steps=1)] == [i for i, _ in extra]
    assert list(s.active) == [0]
    assert s.advance() == []
    tail, = s.advance()
    assert tail["ok"] and tail["token"] == 0 and not s.active
    assert min(allocated_widths) == batch_size // capacity
    assert max(allocated_widths) == batch_size
    q = tail["out"]["joint_positions"]
    names = list(tail["out"]["joint_names"])
    for g in robot.groups:
        np.testing.assert_allclose(
            q[:, [names.index(j) for j in g.joints]], g.home + np.arange(1, 21)[:, None] * 0.001, atol=3e-6
        )


@pytest.mark.parametrize("length", [1, 2, 3])
def test_short_trajectory_loss_is_finite(monkeypatch, length):
    from egowhale.action import base_ik as core

    def fk(kin, q, base, sr, st, group):
        p = q[:, :3, None].transpose(1, 2)
        rotation = torch.zeros((length, 1, 4))
        rotation[:, :, 0] = 1
        return p, rotation

    monkeypatch.setattr(core, "_fk_camera", fk)
    q = torch.zeros((length, 6), requires_grad=True)
    rotation = torch.zeros((length, 1, 4))
    rotation[:, :, 0] = 1
    total, terms = core._loss(
        None,
        q,
        None,
        torch.zeros((length, 1, 3)),
        rotation,
        torch.ones((length, 1)),
        None,
        None,
        None,
        (torch.full((6,), -1.0), torch.ones(6)),
    )
    assert torch.isfinite(total) and all(torch.isfinite(v) for v in terms.values())


def test_actor_returns_short_result_with_long_still_live(monkeypatch):
    from egowhale.batch.actors import ActionActor

    engine = stream()
    actor = ActionActor.__new__(ActionActor)
    actor.steps = {"base_ik": SimpleNamespace(open_stream=lambda size: engine)}
    monkeypatch.setitem(sys.modules, "ray", SimpleNamespace(put=lambda p: p))
    result = actor.advance_ik([dict(ok=True, dst="long", data=20), dict(ok=True, dst="short", data=4)], 324, 0, 2)
    assert result["live"] == ["long"]
    row = result["rows"][0]
    assert row["token"] == "short" and row["save_ref"]["dst"] == "short"
    assert row["save_ref"]["out"] == [1, 2, 3, 4]
    assert set(actor.ik_seconds) == {"long"}
    result = actor.advance_ik([dict(ok=False, dst="failed_read", error="read failed")], 324, 0, 1)
    assert result["rows"][0]["token"] == "failed_read" and result["live"] == ["long"]


def test_jump_repair_preserves_pose_near_joint_limit():
    import mujoco
    from scipy.spatial.transform import Rotation

    from egowhale.action.sequential_ik import _repair_jumps
    from egowhale.robots import load_robot

    robot = load_robot("piper")
    group = robot.groups[0]
    model = robot.model
    data = mujoco.MjData(model)
    joints = [model.joint(name).qposadr[0] for name in group.joints]
    site = model.site(robot.tcp_sites[0]).id
    trajectory = np.tile(group.home, (30, 1))
    trajectory[:15, -1] = np.deg2rad(-179)
    trajectory[15:, -1] = np.deg2rad(179)

    def fk(path):
        positions, rotations = [], []
        for q in path:
            data.qpos[joints] = q
            mujoco.mj_kinematics(model, data)
            positions.append(data.site_xpos[site].copy())
            rotations.append(data.site_xmat[site].reshape(3, 3).copy())
        return np.asarray(positions), np.asarray(rotations)

    p, r = fk(trajectory)
    repaired, before, after, _ = _repair_jumps(
        robot, group, np.eye(4), trajectory, p[:, None], r[:, None], np.ones((30, 1), bool), 18,
    )
    assert (before, after) == (0, 0)
    np.testing.assert_array_equal(repaired, trajectory)
    actual_p, actual_r = fk(repaired)
    assert np.linalg.norm(actual_p - p, axis=1).max() <= 0.01
    assert Rotation.from_matrix(actual_r @ r.transpose(0, 2, 1)).magnitude().max() <= np.deg2rad(10)


def test_periodic_selection_does_not_switch_to_far_elbow_branch():
    from egowhale.action.sequential_ik import _continuous_solution

    positions = torch.deg2rad(torch.tensor([[[-179.], [120.]]]))
    result = SimpleNamespace(js_solution=SimpleNamespace(joint_names=["axis"], position=positions),
                             success=torch.tensor([[True, True]]), position_error=torch.tensor([[.001, .002]]))
    previous = torch.deg2rad(torch.tensor([[179.]]))
    q, error = _continuous_solution(result, previous, torch.tensor([True]), ["axis"], torch.tensor([True]))
    torch.testing.assert_close(q, positions[:, 0])
    assert error.item() == pytest.approx(.001)
    q, _ = _continuous_solution(result, previous, torch.tensor([True]), ["axis"], torch.tensor([False]))
    torch.testing.assert_close(q, positions[:, 1])


def test_angle_lift_is_geometric_and_export_stays_bounded():
    from egowhale.action import base_ik as core

    arm = np.deg2rad([[179., 10.], [-179., 20.], [-175., 30.]])
    mask = np.array([True, False])
    lifted = core._lift_angles(arm, mask)
    np.testing.assert_allclose(np.rad2deg(lifted[:, 0]), [179., 181., 185.])
    np.testing.assert_array_equal(lifted[:, 1], arm[:, 1])
    lo, hi = torch.deg2rad(torch.tensor([-179.9, -30.])), torch.deg2rad(torch.tensor([179.9, 90.]))
    bounded = core._bounded_angles(torch.as_tensor(lifted), lo, hi, torch.as_tensor(mask))
    np.testing.assert_allclose(bounded.numpy(), arm, atol=1e-6)
    assert torch.all(bounded >= lo) and torch.all(bounded <= hi)
    np.testing.assert_allclose(np.rad2deg(core._joint_delta(arm[1:], arm[:-1], mask))[:, 0], [2., 4.])


def test_asset_rotation_mask_excludes_sliders_and_short_ranges():
    import mujoco

    from egowhale.action import base_ik as core

    kinds = [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE, mujoco.mjtJoint.mjJNT_HINGE]
    robot = SimpleNamespace(model=SimpleNamespace(joint=lambda n: SimpleNamespace(type=[kinds[n]])))
    group = SimpleNamespace(joints=[0, 1, 2], lower=np.zeros(3), upper=np.array([2*np.pi, 10., 5.]))
    np.testing.assert_array_equal(core._periodic_joints(robot, group), [True, False, False])


@pytest.fixture
def interval_robot():
    """Small analytical model, independent of repository asset files and CUDA."""
    import mujoco

    model = mujoco.MjModel.from_xml_string("""<mujoco><worldbody><body>
      <joint name="turn" type="hinge" axis="0 0 1" range="-170 170"/>
      <geom type="sphere" size=".01" mass="1"/>
      <site name="tcp_a"/><site name="tcp_b" quat="0 1 0 0"/>
    </body></worldbody></mujoco>""")
    group = SimpleNamespace(name="arm", joints=["turn"], sides=[0, 1],
                            lower=np.deg2rad(np.array([-170.])), upper=np.deg2rad(np.array([170.])))
    robot = SimpleNamespace(model=model, tcp_sites=["tcp_a", "tcp_b"], place_bases=lambda m, b: None)
    return robot, group


def test_local_constraint_repair_checks_both_tcp_and_seams(interval_robot):
    from scipy.spatial.transform import Rotation

    from egowhale.action.sequential_ik import _constrained_interval

    robot, group = interval_robot
    trajectory = np.deg2rad(np.r_[np.full(10, -50.), np.full(10, 50.)])[:, None].astype(np.float32)
    position = np.zeros((20, 2, 3))
    rotation = np.tile(np.eye(3), (20, 2, 1, 1))
    rotation[:, 1] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()
    valid = np.ones((20, 2), bool)
    arm = _constrained_interval(robot, group, np.eye(4), trajectory, trajectory, position, rotation,
                                valid, 4, 16, np.deg2rad(60))
    assert arm is not None and arm.dtype == trajectory.dtype
    joined = np.r_[trajectory[3:4], arm, trajectory[16:17]]
    assert np.max(abs(np.diff(joined, axis=0))) <= np.deg2rad(60) + 1e-6
    assert np.max(abs(arm)) <= np.deg2rad(60)
    # The second TCP tracks the original discontinuous targets exactly. Its
    # 10-degree allowance cannot bridge a 100-degree edge with a 60-degree cap.
    rotation[:, 1] = Rotation.from_rotvec(np.c_[np.zeros((20, 2)), trajectory[:, 0]]).as_matrix() @ rotation[:, 1]
    assert _constrained_interval(robot, group, np.eye(4), trajectory, trajectory, position, rotation,
                                 valid, 4, 16, np.deg2rad(60)) is None


def test_local_constraint_does_not_accumulate_pose_allowance(interval_robot):
    from scipy.spatial.transform import Rotation

    from egowhale.action.sequential_ik import _constrained_interval

    robot, group = interval_robot
    reference = np.deg2rad(np.r_[np.full(10, -50.), np.full(10, 50.)])[:, None]
    rotation = Rotation.from_rotvec(np.c_[np.zeros((20, 2)), reference[:, 0]]).as_matrix()
    rotation = np.stack((rotation, rotation @ Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()), 1)
    current = reference * .85
    # A prior attempt already used 7.5 degrees of the orientation allowance.
    # Referencing current would incorrectly grant a second allowance.
    assert _constrained_interval(robot, group, np.eye(4), current, reference,
                                 np.zeros((20, 2, 3)), rotation, np.ones((20, 2), bool),
                                 4, 16, np.deg2rad(60)) is None


@pytest.mark.parametrize("rotvec", [[0., 0., 0.], [.1, -.2, .3], [1.5, -.6, 1.1]])
def test_rotation_log_world_jacobian_matches_difference(rotvec):
    from scipy.spatial.transform import Rotation

    from egowhale.action.sequential_ik import _rotation_log_jacobian

    rotvec = np.array(rotvec)
    rotation = Rotation.from_rotvec(rotvec)
    eps = 1e-6
    actual = np.column_stack([
        ((Rotation.from_rotvec(axis * eps) * rotation).as_rotvec()
         - (Rotation.from_rotvec(-axis * eps) * rotation).as_rotvec()) / (2 * eps)
        for axis in np.eye(3)
    ])
    np.testing.assert_allclose(_rotation_log_jacobian(rotvec), actual, atol=1e-8, rtol=1e-6)


def test_severe_retry_is_bounded_and_keeps_reference(monkeypatch, interval_robot):
    from egowhale.action import sequential_ik

    robot, group = interval_robot
    q = np.deg2rad(np.r_[np.full(20, -50.), np.full(20, 50.)])[:, None]
    reference = q.copy()
    calls = []

    def retry(robot, group, base, current, ref, position, rotation, valid, start, stop, cap, linear):
        assert ref is reference
        calls.append((start, stop, cap, linear))
        if len(calls) < 5:
            return None
        return np.linspace(current[start-1], current[stop], stop-start+2)[1:-1]

    monkeypatch.setattr(sequential_ik, "_constrained_interval", retry)
    result = sequential_ik._repair_severe_jumps(
        robot, group, np.eye(4), q, reference, np.zeros((40, 2, 3)),
        np.tile(np.eye(3), (40, 2, 1, 1)), np.ones((40, 2), bool), 18,
    )
    assert len(calls) == 5
    assert calls[-1][2] == pytest.approx(np.deg2rad(85))
    assert not np.array_equal(result, q)
    np.testing.assert_array_equal(q, reference)


def test_local_repair_retains_valid_iterate_when_last_step_fails(monkeypatch, interval_robot):
    from scipy.spatial.transform import Rotation

    from egowhale.action import sequential_ik

    robot, group = interval_robot
    q = np.deg2rad(np.r_[np.full(10, -50.), np.full(10, 50.)])[:, None]
    rotations = np.tile(np.eye(3), (20, 2, 1, 1))
    rotations[:, 1] = Rotation.from_rotvec([np.pi, 0, 0]).as_matrix()

    def optimizer(fun, x, **kwargs):
        kwargs["callback"](np.zeros_like(x))  # Valid Cartesian pose and both seams.
        return SimpleNamespace(x=np.full_like(x, np.pi / 2), success=False)

    monkeypatch.setattr(sequential_ik, "minimize", optimizer)
    result = sequential_ik._constrained_interval(
        robot, group, np.eye(4), q, q, np.zeros((20, 2, 3)), rotations,
        np.ones((20, 2), bool), 4, 16, np.deg2rad(60),
    )
    assert result is not None
    np.testing.assert_array_equal(result, np.zeros((12, 1)))


def test_branch_path_pins_seams_and_checks_each_tcp():
    from egowhale.action.sequential_ik import _branch_path
    original = np.array([-.8, -.8, .8, .8, .8])[:, None]
    smooth = np.linspace(-.8, .8, 5)[:, None]
    pool = np.stack([original, smooth], axis=1)
    pe = np.zeros((5, 2, 2))
    re = np.zeros_like(pe)
    reference = np.zeros((5, 2))
    valid = np.ones((5, 2), bool)
    arguments = (reference, reference, valid, np.array([False]), np.array([-1.]), np.array([1.]))
    result = _branch_path(pool, pe, re, *arguments, fixed_first=True, fixed_last=True)
    np.testing.assert_allclose(result, smooth)
    np.testing.assert_array_equal(result[[0, -1]], original[[0, -1]])
    # Only the second TCP fails. Averaging tools would incorrectly admit this pose.
    pe[2, 1, 1] = .003
    result = _branch_path(pool, pe, re, *arguments, fixed_first=True, fixed_last=True)
    np.testing.assert_array_equal(result[2], original[2])
    valid[2, 1] = False
    result = _branch_path(pool, pe, re, *arguments, fixed_first=True, fixed_last=True)
    np.testing.assert_allclose(result, smooth)


def test_branch_path_budget_is_frozen_and_limits_are_not_solver_success():
    from egowhale.action.sequential_ik import _branch_path
    original = np.array([-.8, -.8, .8, .8, .8])[:, None]
    smooth = np.linspace(-.8, .8, 5)[:, None]
    pool = np.stack([original, smooth], axis=1)
    pe = np.full((5, 2, 1), .01)
    re = np.zeros_like(pe)
    ref = np.zeros((5, 1))
    args = (ref, ref, np.ones((5, 1), bool), np.array([False]), np.array([-1.]), np.array([1.]))
    # An earlier repair spent its allowance; comparing to that intermediate would accumulate it.
    result = _branch_path(pool, pe, re, *args, fixed_first=True, fixed_last=True)
    np.testing.assert_array_equal(result, original)
    pe[:] = 0
    pool[2, 1, 0] = 1.1
    result = _branch_path(pool, pe, re, *args, fixed_first=True, fixed_last=True)
    assert result[2, 0] == original[2, 0]
    pool[2, 1, 0] = np.nan
    result = _branch_path(pool, pe, re, *args, fixed_first=True, fixed_last=True)
    assert np.isfinite(result).all()


def test_branch_windows_are_bounded_and_ignore_angle_coordinate_cuts():
    from egowhale.action.sequential_ik import _BRANCH_REGIONS, _branch_windows
    arm = np.zeros((300, 2))
    arm[:, 0] = np.where(np.arange(300) % 2, -np.deg2rad(179), np.deg2rad(179))
    assert _branch_windows(arm, np.array([True, False]), 18) == []
    for frame in range(20, 290, 30):
        arm[frame:, 1] += 1.
    windows = _branch_windows(arm, np.array([True, False]), 18)
    assert len(windows) == _BRANCH_REGIONS
    assert all(0 <= lo < hi <= len(arm) and hi-lo <= 24 for lo, hi in windows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires allocated GPU")
@pytest.mark.parametrize("asset", ["piper", "arx"])
def test_branch_candidate_tail_projection_and_fk_match_mujoco(asset):
    import mujoco
    from curobo.kinematics import Kinematics, KinematicsCfg
    from scipy.spatial.transform import Rotation

    from egowhale.action import base_ik as core
    from egowhale.action.sequential_ik import _BranchRepair
    from egowhale.robots import load_robot

    robot = load_robot(asset)
    group = robot.groups[0]
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(group.config, tool_frames=list(group.tools)))
    repair = _BranchRepair(group, kin)
    q = np.tile(group.home, (2, 1)).astype(np.float32)
    base = np.eye(4)
    pp, qq = repair._poses(q, base)
    position = pp.cpu().numpy()
    quat = qq.cpu().numpy()
    rotation = Rotation.from_quat(quat.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(2, -1, 3, 3)
    # 2*63 rows exercises the fixed-width padded tail, on consecutive cached calls.
    for _ in range(2):
        pool = repair._pool(q, base, position, rotation, core._periodic_joints(robot, group), group.lower, group.upper)
        assert pool.shape == (2, 505, len(group.joints))
        np.testing.assert_array_equal(pool[:, 0], q)
        assert np.isfinite(pool).all()
        assert ((pool >= group.lower-1e-6) & (pool <= group.upper+1e-6)).all()
        pe, re = repair._errors(pool[:, [0, 1, -1]], base, position, rotation)
        robot.place_bases(robot.model, {group.name: base})
        data = mujoco.MjData(robot.model)
        idx = [robot.model.joint(n).qposadr[0] for n in group.joints]
        sites = [robot.model.site(robot.tcp_sites[s]).id for s in group.sides]
        for frame in range(2):
            for column, candidate in enumerate([0, 1, pool.shape[1]-1]):
                data.qpos[idx] = pool[frame, candidate]
                mujoco.mj_kinematics(robot.model, data)
                actual_p = np.linalg.norm(data.site_xpos[sites]-position[frame], axis=-1)
                actual_r = Rotation.from_matrix(data.site_xmat[sites].reshape(-1, 3, 3) @ rotation[frame].transpose(0, 2, 1)).magnitude()
                np.testing.assert_allclose(pe[frame, column], actual_p, atol=1e-5)
                np.testing.assert_allclose(re[frame, column], actual_r, atol=1e-4)
