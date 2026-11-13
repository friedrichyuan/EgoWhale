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

    assert module.episode_capacity("piper", 1296) == 8
    assert module.episode_capacity("arx", 1296) == 16
    with pytest.raises(ValueError):
        module.episode_capacity("piper", 200)
    p = tmp_path / "c.yaml"
    p.write_text("robot: piper\nepisodes: a.hdf5\nnodes: {action: {batch_size: 1296}}\nmemory: {inflight: 7}\n")
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

    class Solver:
        def __init__(self, cfg):
            self.group = groups[id(cfg["robot"])]
            self.joint_names = list(self.group.joints)[::-1]
            self.width = cfg["max_batch_size"]

        def solve_pose(self, goal_tool_poses, current_state):
            # Model a cuRobo solver that reuses buffers sized for the configured batch.
            assert len(current_state.position) == self.width
            for pose in goal_tool_poses.values():
                assert len(pose.position) == len(pose.quaternion) == self.width
                torch.testing.assert_close(pose.quaternion.norm(dim=-1), torch.ones(self.width))
            values = current_state.position + 0.001
            # Exercise returned joint order differing from solver order.
            return SimpleNamespace(
                js_solution=SimpleNamespace(position=values.flip(-1)[:, None], joint_names=list(self.group.joints)),
                position_error=torch.zeros((len(values), 1)),
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
    def refine(kin, p, r, v, sr, st, o, a, g, fps):
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
