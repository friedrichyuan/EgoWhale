"""Joint base pose and arm trajectory, seeded by cuRobo IK and refined with smoothness."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from egowhale.step import BASE, GRIPPER, IK, ROOT, Step

ARM = (
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "right_joint11",
    "right_joint12",
    "right_joint13",
    "right_joint14",
    "right_joint15",
    "right_joint16",
)
GRIPPERS = (("left_joint7", "left_joint8"), ("right_joint17", "right_joint18"))
TCP = ("left_tcp", "right_tcp")
SCENE = ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"
URDF = ROOT / "assets" / "curobo_arx" / "arx_acone_kin.urdf"
YAML = ROOT / "assets" / "curobo_arx" / "arx_acone.yml"

# OpenCV camera (x right, y down, z forward) from robot (x forward, y left, z up).
_R_CAM_ROBOT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)
_REACH = 0.855
_DT = 1.0 / 30.0
_BOUND = np.array([0.30, 0.30, 0.20, np.deg2rad(35.0)], dtype=np.float64)
_SEEDS = 2
_STEPS = 120


class BaseIK(Step):
    name = "base_ik"
    needs = (GRIPPER,)
    makes = (BASE, IK)
    gpus = 1

    def run(self, src: Path, dst: Path) -> None:
        result = self.compute(read_gripper(Path(dst)))
        save_base(Path(dst), result)
        losses = result["losses"]
        print(f"  pos {losses['position'] * 1000:.1f} mm  jerk {losses['jerk']:.4f}")

    def compute(self, data: dict) -> dict:
        matrix, qpos, losses = _solve(data["position"], data["rotation"], data["width"], data["valid"])
        return {"matrix": matrix, "qpos": qpos.astype(np.float32), "losses": losses}


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
    base_path.write_text(json.dumps({"matrix": result["matrix"].tolist(), "losses": result["losses"]}, indent=2))
    np.savez_compressed(dst / IK, qpos=result["qpos"])


def _solve(position, rotation, width, valid):
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.kinematics import Kinematics, KinematicsCfg

    seed_r, seed_t = _seed_pose(position, valid)
    goal_p, goal_r = _hold(position, rotation, valid)
    offsets = _offset_grid()
    robot = _robot_yaml()
    kin = Kinematics(KinematicsCfg.from_robot_yaml_file(robot, tool_frames=list(TCP)))
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
    offset, arm = _best_seed(ik, goal_p, goal_r, seed_r, seed_t, offsets)
    offset, arm, losses = _refine(kin, goal_p, goal_r, valid, seed_r, seed_t, offset, arm)
    arm = _smooth_arm(arm)
    qpos = _pack_qpos(arm, width, valid)
    return _camera_matrix(seed_r, seed_t, offset), qpos, losses


def _smooth_arm(arm: np.ndarray) -> np.ndarray:
    window = min(15, arm.shape[0] if arm.shape[0] % 2 else arm.shape[0] - 1)
    if window <= 3:
        return arm
    return savgol_filter(arm, window, 3, axis=0)


def _offset_grid() -> np.ndarray:
    samples = [
        np.linspace(-bound, bound, 3)
        for bound in _BOUND
    ]
    grid = np.stack(np.meshgrid(*samples, indexing="ij"), axis=-1).reshape(-1, 4)
    return grid.astype(np.float64)


def _best_seed(ik, position, rotation, seed_r, seed_t, offsets):
    frames = len(position)
    posed = [
        _into_base(position, rotation, _camera_matrix(seed_r, seed_t, offset))
        for offset in offsets
    ]
    stacked_p = np.concatenate([item[0] for item in posed], axis=0)
    stacked_r = np.concatenate([item[1] for item in posed], axis=0)
    arm, error = _batch_ik(ik, stacked_p, stacked_r)
    arm = arm.reshape(len(offsets), frames, -1)
    error = error.reshape(len(offsets), frames, -1)
    choice = int(np.nanmean(error, axis=(1, 2)).argmin())
    return offsets[choice], arm[choice]


def _refine(kin, position, rotation, valid, seed_r, seed_t, offset, arm):
    pos, quat = _wxyz_pair(position, rotation)
    goal_p = torch.tensor(pos, device="cuda", dtype=torch.float32)
    goal_q = torch.tensor(quat, device="cuda", dtype=torch.float32)
    mask = torch.tensor(valid, device="cuda", dtype=torch.float32)
    seed_R = torch.tensor(seed_r, device="cuda", dtype=torch.float32)
    seed_t = torch.tensor(seed_t, device="cuda", dtype=torch.float32)
    lower, upper = _limits()
    q = torch.nn.Parameter(torch.tensor(arm, device="cuda", dtype=torch.float32))
    base = torch.nn.Parameter(torch.tensor(offset, device="cuda", dtype=torch.float32))
    opt = torch.optim.Adam([{"params": [base], "lr": 1e-2}, {"params": [q], "lr": 2e-2}])
    bounds = torch.tensor(_BOUND, device="cuda", dtype=torch.float32)
    for _step in range(_STEPS):
        opt.zero_grad(set_to_none=True)
        total, _terms = _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t)
        if not torch.isfinite(total):
            break
        total.backward()
        torch.nn.utils.clip_grad_norm_([q, base], 5.0)
        opt.step()
        with torch.no_grad():
            q.clamp_(lower, upper)
            base.clamp_(-bounds, bounds)
    with torch.no_grad():
        _total, terms = _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t)
    losses = {key: float(value.detach().cpu()) for key, value in terms.items()}
    return base.detach().cpu().numpy(), q.detach().cpu().numpy(), losses


def _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t):
    pos, quat = _fk_camera(kin, q, base, seed_R, seed_t)
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
    low, high = _limits()
    joint_limit = torch.relu(low - q).square().mean() + torch.relu(q - high).square().mean()
    total = 20.0 * position + 5.0 * orientation + 4.0 * continuity + 0.05 * velocity + 2.0 * jerk + 5.0 * joint_limit
    return total, {
        "position": position,
        "orientation": orientation,
        "continuity": continuity,
        "velocity": velocity,
        "jerk": jerk,
    }


def _fk_camera(kin, q, base, seed_R, seed_t):
    from curobo._src.geom.transform import matrix_to_quaternion
    from curobo.types import JointState

    names = list(kin.joint_names)
    ordered = q if names == list(ARM) else q[:, [list(ARM).index(name) for name in names]]
    ordered = ordered.reshape(ordered.shape[0], 1, ordered.shape[-1]).contiguous()
    state = kin.compute_kinematics(JointState.from_position(ordered, joint_names=names))
    frames = list(state.tool_poses.tool_frames)
    order = [frames.index(name) for name in TCP]
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


def _batch_ik(ik, position, rotation):
    from curobo.types import GoalToolPose, Pose

    pos, quat = _wxyz_pair(position, rotation)
    poses = {
        name: Pose(
            position=torch.tensor(pos[:, side], device="cuda", dtype=torch.float32),
            quaternion=torch.tensor(quat[:, side], device="cuda", dtype=torch.float32),
            name=name,
            normalize_rotation=True,
        )
        for side, name in enumerate(TCP)
    }
    result = ik.solve_pose(goal_tool_poses=GoalToolPose.from_poses(poses, ordered_tool_frames=list(TCP)))
    return _arm_from_result(result, ik)[: len(pos)], _frame_error(result, len(pos))


def _arm_from_result(result, ik) -> np.ndarray:
    state = result.js_solution
    values = state.position.detach().float().cpu().numpy()
    while values.ndim > 2:
        values = values[:, 0]
    names = list(state.joint_names or ik.joint_names)
    index = [names.index(name) for name in ARM]
    return np.asarray(values[:, index], dtype=np.float64)


def _frame_error(result, width: int) -> np.ndarray:
    error = getattr(result, "position_error", None)
    if error is None:
        return np.zeros((width, 1), dtype=np.float64)
    values = error.detach().float().cpu().numpy().reshape(error.shape[0], -1)[:width]
    return np.asarray(values, dtype=np.float64)


def _limits():
    if not hasattr(_limits, "pair"):
        model = mujoco.MjModel.from_xml_path(str(SCENE))
        low, high = [], []
        for name in ARM:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            low.append(model.jnt_range[joint, 0])
            high.append(model.jnt_range[joint, 1])
        _limits.pair = (
            torch.tensor(low, device="cuda", dtype=torch.float32),
            torch.tensor(high, device="cuda", dtype=torch.float32),
        )
    return _limits.pair


def _pack_qpos(arm, width, valid) -> np.ndarray:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    qpos = np.zeros((len(arm), model.nq), dtype=np.float64)
    for column, name in enumerate(ARM):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qpos[:, model.jnt_qposadr[joint]] = arm[:, column]
    gaps = _gaps(model)
    for side, names in enumerate(GRIPPERS):
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        upper = min(float(model.jnt_range[joint, 1]) for joint in joints)
        opening = np.clip(0.5 * (width[:, side] - gaps[side]), 0.0, upper)
        opening = np.where(valid[:, side], opening, 0.0)
        for joint in joints:
            qpos[:, model.jnt_qposadr[joint]] = opening
    return qpos


def _gaps(model) -> list[float]:
    data = mujoco.MjData(model)
    gaps = []
    for names in GRIPPERS:
        data.qpos[:] = 0.0
        bodies = []
        for name in names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = model.jnt_range[joint, 0]
            bodies.append(int(model.jnt_bodyid[joint]))
        mujoco.mj_forward(model, data)
        gaps.append(float(np.linalg.norm(data.xpos[bodies[0]] - data.xpos[bodies[1]])))
    return gaps


def _seed_pose(position, valid):
    usable = valid.all(axis=1) if valid.all(axis=1).any() else valid.any(axis=1)
    midpoint = 0.5 * (position[usable, 0] + position[usable, 1]).mean(axis=0)
    translation = midpoint - _R_CAM_ROBOT @ np.array([0.65 * _REACH, 0.0, 0.15])
    return _R_CAM_ROBOT, translation


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
    for side in range(2):
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


def _robot_yaml() -> str:
    text = YAML.read_text(encoding="utf-8")
    text = text.replace('urdf_path: "arx_acone_kin.urdf"', f'urdf_path: "{URDF}"')
    text = text.replace('asset_root_path: ""', f'asset_root_path: "{URDF.parent}"')
    path = Path(tempfile.gettempdir()) / "egowhale_arx.yml"
    path.write_text(text, encoding="utf-8")
    return str(path)
