"""Move from the zero configuration to the first frame with cuRobo TrajOpt."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from egowhale.media import FPS
from egowhale.paths import IK, PREFIX
from egowhale.robots import load_robot
from egowhale.robots.motion import read_motion
from egowhale.step import Step


class Approach(Step):
    name = "approach"
    needs = (IK,)
    makes = (PREFIX,)
    gpus = 1

    def __init__(self, robot="arx"):
        super().__init__()
        self.robot = load_robot(robot)

    def run(self, src: Path, dst: Path) -> None:
        prefix = self.compute(self.read(src, dst))
        self.save(dst, prefix)
        print(f"  approach {len(prefix['joint_positions'])} frames  {(len(prefix['joint_positions']) - 1) / FPS:.2f}s")

    def read(self, src: Path, dst: Path) -> dict:
        return read_ik(Path(dst))

    def save(self, dst: Path, out: dict) -> None:
        save_prefix(Path(dst), out)

    def compute(self, data: dict) -> dict:
        motion = data if "joint_positions" in data else self.robot.named(data["qpos"])
        names = list(motion["joint_names"])
        q = np.asarray(motion["joint_positions"], dtype=np.float64)
        paths = []
        for group in self.robot.groups:
            columns = [names.index(n) for n in group.joints]
            goal = q[0, columns]
            step = q[min(1, len(q) - 1), columns] - goal
            paths.append((group, _trajopt(goal, group), goal, step))
        count = max(len(path) for _, path, _, _ in paths)
        result = np.zeros((count, len(names)), dtype=np.float32)
        arm_names = set()
        for group, path, goal, step in paths:
            if len(path) != count:
                path = np.stack([np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(path)), col) for col in path.T], axis=1)
            columns = [names.index(n) for n in group.joints]
            result[:, columns] = _match_arrival(path, goal, step, origin=group.home)
            arm_names.update(group.joints)
        columns = [i for i, n in enumerate(names) if n not in arm_names]
        goal = q[0, columns]
        step = q[min(1, len(q) - 1), columns] - goal
        result[:, columns] = _match_arrival(_ease(count) * goal, goal, step)
        return {"joint_names": np.asarray(names), "joint_positions": result, "fps": np.float32(FPS)}


def read_ik(dst: Path) -> dict:
    return read_motion(Path(dst) / IK)


def save_prefix(dst: Path, result: dict) -> None:
    path = Path(dst) / PREFIX
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **result)


def _trajopt(goal: np.ndarray, group) -> np.ndarray:
    from curobo.trajectory_optimizer import TrajectoryOptimizer, TrajectoryOptimizerCfg
    from curobo.types import JointState

    torch.set_default_dtype(torch.float32)
    solver = TrajectoryOptimizer(TrajectoryOptimizerCfg.create(
        robot=group.config,
        num_seeds=4,
        self_collision_check=False,
        load_collision_spheres=False,
        use_cuda_graph=False,
        max_batch_size=1,
    ))
    names = list(solver.joint_names)
    ordered = torch.tensor(goal[[list(group.joints).index(name) for name in names]], device="cuda", dtype=torch.float32).view(1, -1)
    result = solver.solve_cspace(
        goal_state=JointState.from_position(ordered, joint_names=names),
        current_state=JointState.from_position(torch.tensor(group.home[[list(group.joints).index(n) for n in names]], device="cuda", dtype=torch.float32).view(1, -1), joint_names=names),
    )
    position = result.js_solution.position.detach().float().cpu().numpy()
    while position.ndim > 2:
        position = position[0]
    dt = float(result.js_solution.dt.detach().float().cpu().reshape(-1)[0])
    samples = np.arange(len(position)) * dt
    query = np.linspace(0.0, samples[-1], max(2, int(round(samples[-1] * FPS)) + 1))
    arm = np.stack([np.interp(query, samples, position[:, index]) for index in range(position.shape[1])], axis=1)
    arm = arm[:, [names.index(name) for name in group.joints]]
    arm[-1] = goal
    if np.linalg.norm(position[-1] - ordered.detach().cpu().numpy().reshape(-1)) > 1e-3:
        raise RuntimeError("approach did not reach the first frame")
    return arm


def _ease(count: int) -> np.ndarray:
    progress = np.linspace(0.0, 1.0, count)[:, None]
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


def _match_arrival(path: np.ndarray, goal: np.ndarray, end_step: np.ndarray, tail: int = 8, origin=None) -> np.ndarray:
    """Quintic the last samples so the prefix arrives on the first IK frame and its next step."""
    path = np.array(path, dtype=np.float64, copy=True)
    tail = min(tail, len(path) - 1)
    start = len(path) - 1 - tail
    initial = np.zeros_like(goal) if origin is None else np.asarray(origin)
    origin = path[start]
    start_step = path[start] - path[start - 1] if start > 0 else np.zeros_like(origin)
    fraction = np.linspace(0.0, 1.0, tail + 1)[:, None]
    span = float(tail)
    h00 = 2.0 * fraction**3 - 3.0 * fraction**2 + 1.0
    h10 = fraction**3 - 2.0 * fraction**2 + fraction
    h01 = -2.0 * fraction**3 + 3.0 * fraction**2
    h11 = fraction**3 - fraction**2
    path[start:] = h00 * origin + h10 * (start_step * span) + h01 * goal + h11 * (end_step * span)
    path[0] = initial
    path[-1] = goal
    return path
