"""Move from the zero configuration to the first frame with cuRobo TrajOpt."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch

from egowhale.action.base_ik import ARM, GRIPPERS, SCENE, _robot_yaml
from egowhale.step import IK, PREFIX, Step

_FPS = 30.0


class Approach(Step):
    name = "approach"
    needs = (IK,)
    makes = (PREFIX,)
    gpus = 1

    def run(self, src: Path, dst: Path) -> None:
        prefix = self.compute(read_ik(Path(dst)))
        save_prefix(Path(dst), prefix)
        print(f"  approach {len(prefix['qpos'])} frames  {(len(prefix['qpos']) - 1) / _FPS:.2f}s")

    def compute(self, data: dict) -> dict:
        qpos = data["qpos"]
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        goal = _read(model, ARM, qpos[0])
        arm_step = _read(model, ARM, qpos[1]) - goal
        grip_names = _gripper_names()
        grip_goal = _read(model, grip_names, qpos[0])
        grip_step = _read(model, grip_names, qpos[1]) - grip_goal
        arm = _match_arrival(_trajopt(goal), goal, arm_step)
        grip = _match_arrival(_ease(len(arm)) * grip_goal, grip_goal, grip_step)
        prefix = np.zeros((len(arm), model.nq), dtype=np.float32)
        _write(model, ARM, arm, prefix)
        _write(model, _gripper_names(), grip, prefix)
        return {"qpos": prefix}


def read_ik(dst: Path) -> dict:
    return {"qpos": np.load(Path(dst) / IK)["qpos"]}


def save_prefix(dst: Path, result: dict) -> None:
    path = Path(dst) / PREFIX
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, qpos=result["qpos"])


def _trajopt(goal: np.ndarray) -> np.ndarray:
    from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
    from curobo.types import JointState

    torch.set_default_dtype(torch.float32)
    solver = TrajectoryOptimizer(TrajectoryOptimizerCfg.create(
        robot=_robot_yaml(),
        num_seeds=4,
        self_collision_check=False,
        load_collision_spheres=False,
        use_cuda_graph=False,
        max_batch_size=1,
    ))
    names = list(solver.joint_names)
    ordered = torch.tensor(goal[[list(ARM).index(name) for name in names]], device="cuda", dtype=torch.float32).view(1, -1)
    result = solver.solve_cspace(
        goal_state=JointState.from_position(ordered, joint_names=names),
        current_state=JointState.from_position(torch.zeros_like(ordered), joint_names=names),
    )
    position = result.js_solution.position.detach().float().cpu().numpy()
    while position.ndim > 2:
        position = position[0]
    dt = float(result.js_solution.dt.detach().float().cpu().reshape(-1)[0])
    samples = np.arange(len(position)) * dt
    query = np.linspace(0.0, samples[-1], max(2, int(round(samples[-1] * _FPS)) + 1))
    arm = np.stack([np.interp(query, samples, position[:, index]) for index in range(position.shape[1])], axis=1)
    arm = arm[:, [names.index(name) for name in ARM]]
    arm[-1] = goal
    if np.linalg.norm(position[-1] - ordered.detach().cpu().numpy().reshape(-1)) > 1e-3:
        raise RuntimeError("approach did not reach the first frame")
    return arm


def _ease(count: int) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, count)[:, None]
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def _match_arrival(path: np.ndarray, goal: np.ndarray, end_step: np.ndarray, tail: int = 8) -> np.ndarray:
    """Quintic the last samples so the prefix arrives on the first IK frame and its next step."""
    path = np.array(path, dtype=np.float64, copy=True)
    tail = min(tail, len(path) - 1)
    start = len(path) - 1 - tail
    origin = path[start]
    start_step = path[start] - path[start - 1] if start > 0 else np.zeros_like(origin)
    fraction = np.linspace(0.0, 1.0, tail + 1)[:, None]
    span = float(tail)
    h00 = 2.0 * fraction**3 - 3.0 * fraction**2 + 1.0
    h10 = fraction**3 - 2.0 * fraction**2 + fraction
    h01 = -2.0 * fraction**3 + 3.0 * fraction**2
    h11 = fraction**3 - fraction**2
    path[start:] = h00 * origin + h10 * (start_step * span) + h01 * goal + h11 * (end_step * span)
    path[0] = 0.0
    path[-1] = goal
    return path


def _gripper_names() -> tuple[str, ...]:
    return tuple(name for pair in GRIPPERS for name in pair)


def _read(model, names, qpos: np.ndarray) -> np.ndarray:
    return np.array([
        qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]
        for name in names
    ], dtype=np.float64)


def _write(model, names, values: np.ndarray, qpos: np.ndarray) -> None:
    for column, name in enumerate(names):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[:, model.jnt_qposadr[joint]] = values[:, column]
