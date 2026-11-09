"""Joint base pose and arm trajectory, seeded by cuRobo IK and refined with smoothness."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from egowhale.media import FPS
from egowhale.paths import BASE, GRIPPER, IK
from egowhale.robots import load_robot
from egowhale.step import Step

_DT = 1.0 / FPS
_SEEDS = 2
_STEPS = 120


class BaseIK(Step):
    name = "base_ik"
    needs = (GRIPPER,)
    makes = (BASE, IK)
    gpus = 1

    def __init__(self, robot="arx"):
        super().__init__()
        self.robot = load_robot(robot)

    def run(self, src: Path, dst: Path) -> None:
        result = self.compute(self.read(src, dst))
        self.save(dst, result)
        losses = result["losses"]
        print(f"  pos {losses['position'] * 1000:.1f} mm  jerk {losses['jerk']:.4f}")

    def read(self, src: Path, dst: Path) -> dict:
        return read_gripper(Path(dst))

    def save(self, dst: Path, out: dict) -> None:
        save_base(Path(dst), out)

    def compute(self, data: dict) -> dict:
        arms, bases, losses = [], {}, {}
        for group in self.robot.groups:
            sides = list(group.sides)
            matrix, arm, terms = _solve(data["position"][:, sides], data["rotation"][:, sides],
                                        data["valid"][:, sides], group)
            arms.append(arm)
            bases[group.name] = matrix
            losses[group.name] = terms
        motion = self.robot.pack(arms, data["width"], data["valid"])
        # Keep the existing console summary; per-group losses are saved separately.
        summary = {key: float(np.mean([v[key] for v in losses.values()])) for key in next(iter(losses.values()))}
        return {**motion, "bases": bases, "losses": summary, "group_losses": losses}


def read_gripper(dst: Path) -> dict:
    with np.load(Path(dst) / GRIPPER) as data:
        return {
            "position": np.asarray(data["position"], dtype=np.float64),
            "rotation": np.asarray(data["rotation"], dtype=np.float64),
            "width": np.asarray(data["width"], dtype=np.float64),
            "valid": np.asarray(data["valid"], dtype=bool),
        }


def save_base(dst: Path, result: dict) -> None:
    dst = Path(dst)
    base_path = dst / BASE
    base_path.parent.mkdir(parents=True, exist_ok=True)
    base_path.write_text(json.dumps({"bases": {n: m.tolist() for n, m in result["bases"].items()}, "losses": result["losses"], "group_losses": result["group_losses"]}, indent=2))
    np.savez_compressed(dst / IK, joint_names=result["joint_names"], joint_positions=result["joint_positions"], fps=np.float32(FPS))


def _solve(position, rotation, valid, group):
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.kinematics import Kinematics, KinematicsCfg

    seed_r, seed_t = group.seed(position, valid)
    goal_p, goal_r = _hold(position, rotation, valid)
    offsets = _offset_grid(group.bounds)
    robot = group.config
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot, tool_frames=list(group.tools)))
    ik = InverseKinematics(
        InverseKinematicsCfg.create(
            robot=robot,
            num_seeds=_SEEDS,
            self_collision_check=False,
            load_collision_spheres=False,
            use_cuda_graph=False,
            max_batch_size=len(offsets) * len(goal_p),
            optimizer_configs=["ik/lbfgs_ik.yml"],
        )
    )
    offset, arm = _best_seed(ik, goal_p, goal_r, seed_r, seed_t, offsets, group)
    offset, arm, losses = _refine(kin, goal_p, goal_r, valid, seed_r, seed_t, offset, arm, group)
    arm = _smooth_arm(arm)
    return _camera_matrix(seed_r, seed_t, offset), arm, losses


def _smooth_arm(arm: np.ndarray) -> np.ndarray:
    window = min(15, arm.shape[0] if arm.shape[0] % 2 else arm.shape[0] - 1)
    if window <= 3:
        return arm
    return savgol_filter(arm, window, 3, axis=0)


def _offset_grid(bounds) -> np.ndarray:
    samples = [
        np.linspace(-bound, bound, 3)
        for bound in bounds
    ]
    grid = np.stack(np.meshgrid(*samples, indexing="ij"), axis=-1).reshape(-1, 4)
    return grid.astype(np.float64)


def _best_seed(ik, position, rotation, seed_r, seed_t, offsets, group):
    frames = len(position)
    posed = [
        _into_base(position, rotation, _camera_matrix(seed_r, seed_t, offset))
        for offset in offsets
    ]
    stacked_p = np.concatenate([item[0] for item in posed], axis=0)
    stacked_r = np.concatenate([item[1] for item in posed], axis=0)
    arm, error = _batch_ik(ik, stacked_p, stacked_r, group)
    arm = arm.reshape(len(offsets), frames, -1)
    error = error.reshape(len(offsets), frames, -1)
    choice = int(np.nanmean(error, axis=(1, 2)).argmin())
    return offsets[choice], arm[choice]


def _refine(kin, position, rotation, valid, seed_r, seed_t, offset, arm, group):
    pos, quat = _wxyz_pair(position, rotation)
    goal_p = torch.tensor(pos, device="cuda", dtype=torch.float32)
    goal_q = torch.tensor(quat, device="cuda", dtype=torch.float32)
    mask = torch.tensor(valid, device="cuda", dtype=torch.float32)
    seed_R = torch.tensor(seed_r, device="cuda", dtype=torch.float32)
    seed_t = torch.tensor(seed_t, device="cuda", dtype=torch.float32)
    lower, upper = _limits(group)
    q = torch.nn.Parameter(torch.tensor(arm, device="cuda", dtype=torch.float32))
    base = torch.nn.Parameter(torch.tensor(offset, device="cuda", dtype=torch.float32))
    opt = torch.optim.Adam([{"params": [base], "lr": 1e-2}, {"params": [q], "lr": 2e-2}])
    bounds = torch.tensor(group.bounds, device="cuda", dtype=torch.float32)
    for _step in range(_STEPS):
        opt.zero_grad(set_to_none=True)
        total, _terms = _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t, group, (lower, upper))
        if not torch.isfinite(total):
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_([q, base], 5.0)
        opt.step()
        with torch.no_grad():
            q.clamp_(lower, upper)
            base.clamp_(-bounds, bounds)
    with torch.no_grad():
        _total, terms = _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t, group, (lower, upper))
    losses = {key: float(value.detach().cpu()) for key, value in terms.items()}
    return base.detach().cpu().numpy(), q.detach().cpu().numpy(), losses


def _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t, group, limits):
    pos, quat = _fk_camera(kin, q, base, seed_R, seed_t, group)
    pos_err = torch.linalg.norm(pos - goal_p, dim=-1)
    dot = (quat * goal_q).sum(dim=-1).abs().clamp(0.0, 1.0)
    ori = 1.0 - dot.square()
    scale = mask.sum().clamp_min(1.0)
    position = (pos_err * mask).sum() / scale
    orientation = (ori * mask).sum() / scale
    delta = q[1:] - q[:-1]
    continuity = delta.square().mean()
    velocity = (delta / _DT).square().mean()
    jerk = (q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]).square().mean()
    low, high = limits
    joint_limit = torch.relu(low - q).square().mean() + torch.relu(q - high).square().mean()
    total = 20.0 * position + 5.0 * orientation + 4.0 * continuity + 0.05 * velocity + 2.0 * jerk + 5.0 * joint_limit
    return total, {
        "position": position,
        "orientation": orientation,
        "continuity": continuity,
        "velocity": velocity,
        "jerk": jerk,
    }


def _fk_camera(kin, q, base, seed_R, seed_t, group):
    from curobo._src.geom.transform import matrix_to_quaternion
    from curobo.types import JointState

    names = list(kin.joint_names)
    ordered = q if names == list(group.joints) else q[:, [list(group.joints).index(name) for name in names]]
    ordered = ordered.reshape(ordered.shape[0], 1, ordered.shape[-1]).contiguous()
    state = kin.compute_kinematics(JointState.from_position(ordered, joint_names=names))
    frames = list(state.tool_poses.tool_frames)
    order = [frames.index(name) for name in group.tools]
    pos = state.tool_poses.position[:, 0, order].float()
    quat = state.tool_poses.quaternion[:, 0, order].float()
    yaw = base[3]
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    zeros = torch.zeros((), device=q.device, dtype=torch.float32)
    rotor = torch.stack((
        torch.stack((cosine, -sine, zeros)),
        torch.stack((sine, cosine, zeros)),
        torch.stack((zeros, zeros, torch.ones((), device=q.device, dtype=torch.float32))),
    ))
    rotation = (seed_R.float() @ rotor).float()
    translation = seed_t.float() + seed_R.float() @ base[:3].float()
    pos = torch.einsum("ij,tlj->tli", rotation, pos) + translation
    quat = _quat_mul(matrix_to_quaternion(rotation).reshape(4), quat)
    quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return pos, quat


def _quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), -1)


def _batch_ik(ik, position, rotation, group):
    from curobo.types import GoalToolPose, Pose

    pos, quat = _wxyz_pair(position, rotation)
    poses = {
        name: Pose(
            position=torch.tensor(pos[:, side], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor(quat[:, side], device="cuda", dtype=torch.float32),
            name=name,
            normalize_rotation=True,
        )
        for side, name in enumerate(group.tools)
    }
    result = ik.solve_pose(goal_tool_poses=GoalToolPose.from_poses(poses, ordered_tool_frames=list(group.tools)))
    return _arm_from_result(result, ik, group)[: len(pos)], _frame_error(result, len(pos))


def _arm_from_result(result, ik, group) -> np.ndarray:
    state = result.js_solution
    values = state.position.detach().float().cpu().numpy()
    while values.ndim > 2:
        values = values[:, 0]
    names = list(state.joint_names or ik.joint_names)
    index = [names.index(name) for name in group.joints]
    return np.asarray(values[:, index], dtype=np.float64)


def _frame_error(result, width: int) -> np.ndarray:
    error = getattr(result, "position_error", None)
    if error is None:
        return np.zeros((width, 1), dtype=np.float64)
    values = error.detach().float().cpu().numpy().reshape(error.shape[0], -1)[:width]
    return np.asarray(values, dtype=np.float64)


def _limits(group):
    return (torch.as_tensor(group.lower, device="cuda", dtype=torch.float32),
            torch.as_tensor(group.upper, device="cuda", dtype=torch.float32))


def _camera_matrix(seed_r, seed_t, offset) -> np.ndarray:
    dx, dy, dz, yaw = np.asarray(offset, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = seed_r @ Rotation.from_euler("z", float(yaw)).as_matrix()
    transform[:3, 3] = seed_t + seed_r @ np.array([dx, dy, dz])
    return transform


def _into_base(position, rotation, transform):
    inverse_r = transform[:3, :3].T
    inverse_t = -inverse_r @ transform[:3, 3]
    pos = np.einsum("ij,...j->...i", inverse_r, position) + inverse_t
    rot = np.einsum("ij,...jk->...ik", inverse_r, rotation)
    return pos, rot


def _hold(position, rotation, valid):
    position, rotation = position.copy(), rotation.copy()
    for side in range(position.shape[1]):
        good = np.flatnonzero(valid[:, side])
        if len(good) == 0:
            position[:, side] = 0.0
            rotation[:, side] = np.eye(3)
            continue
        last = position[good[0], side], rotation[good[0], side]
        for frame in range(len(position)):
            if valid[frame, side]:
                last = position[frame, side].copy(), rotation[frame, side].copy()
            else:
                position[frame, side], rotation[frame, side] = last
    return position, rotation


def _wxyz_pair(position, rotation):
    xyzw = Rotation.from_matrix(rotation.reshape(-1, 3, 3)).as_quat()
    quat = xyzw.reshape(rotation.shape[:-2] + (4,))[..., [3, 0, 1, 2]]
    return position, quat
