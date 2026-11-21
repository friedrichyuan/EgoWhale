"""Joint base pose and arm trajectory, seeded by cuRobo IK and refined with smoothness."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from egowhale.media import FPS, data_fps, valid_fps, video_shape
from egowhale.paths import BASE, GRIPPER, IK
from egowhale.robots import load_robot
from egowhale.step import Step

_SEEDS = 2
_STEPS = 120
_POSITION_WEIGHT = 20.0
_ORIENTATION_WEIGHT = 5.0


class BaseIK(Step):
    name = "base_ik"
    needs = (GRIPPER,)
    makes = (BASE, IK)
    gpus = 1

    def __init__(self, robot="arx", ik_method="base_ik", batch_size=0):
        super().__init__()
        self.robot = load_robot(robot)
        self.ik_method = ik_method
        self.batch_size = batch_size

    def run(self, src: Path, dst: Path) -> None:
        result = self.compute(self.read(src, dst))
        self.save(dst, result)
        losses = result["losses"]
        print(f"  pos {losses['position'] * 1000:.1f} mm  jerk {losses['jerk']:.4f}")

    def read(self, src: Path, dst: Path) -> dict:
        data = read_gripper(Path(dst))
        if Path(src).with_suffix(".mp4").is_file():
            data["camera_size"] = camera_size(Path(src))
        return data

    def save(self, dst: Path, out: dict) -> None:
        save_base(Path(dst), out)

    def open_stream(self, batch_size=None):
        from egowhale.action.base_ik_seq import EpisodeStream
        return EpisodeStream(self.robot, self.batch_size if batch_size is None else batch_size)

    def compute_batch(self, items: list[dict], batch_size: int) -> list[dict]:
        from egowhale.action.base_ik_seq import compute_batch
        return compute_batch(self.robot, items, batch_size)

    @property
    def base_bounds(self):
        if not hasattr(self, "_base_bounds"):
            self._base_bounds = base_boxes(self.robot)
        return self._base_bounds

    def compute(self, data: dict) -> dict:
        if self.ik_method == "base_ik_seq":
            return self.compute_batch([data], self.batch_size)[0]
        arms, bases, losses = [], {}, {}
        for group in self.robot.groups:
            sides = list(group.sides)
            options = {"periodic": _periodic_joints(self.robot, group)}
            if "camera_size" in data:
                options["geometry"] = (self.base_bounds.get(group.name), data["intrinsic"], data["camera_size"])
            matrix, arm, terms = _solve(data["position"][:, sides], data["rotation"][:, sides],
                                        data["valid"][:, sides], group, fps=data_fps(data), **options)
            arms.append(arm)
            bases[group.name] = matrix
            losses[group.name] = terms
        motion = self.robot.pack(arms, data["width"], data["valid"])
        # Keep the existing console summary; per-group losses are saved separately.
        summary = {key: float(np.mean([v[key] for v in losses.values()])) for key in next(iter(losses.values()))}
        return {**motion, "fps": data_fps(data), "bases": bases, "losses": summary, "group_losses": losses}


def read_gripper(dst: Path) -> dict:
    with np.load(Path(dst) / GRIPPER) as data:
        return {
            "fps": data_fps(data),
            "intrinsic": np.asarray(data["intrinsic"]) if "intrinsic" in data else None,
            "position": np.asarray(data["position"], dtype=np.float64),
            "rotation": np.asarray(data["rotation"], dtype=np.float64),
            "width": np.asarray(data["width"], dtype=np.float64),
            "valid": np.asarray(data["valid"], dtype=bool),
        }


def save_base(dst: Path, result: dict) -> None:
    dst = Path(dst)
    base_path = dst / BASE
    base_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"bases": {n: m.tolist() for n, m in result["bases"].items()}, "losses": result["losses"],
              "group_losses": result["group_losses"]}
    for name in ("base_frame", "pair_constraints_met", "motion_model"):
        if name in result:
            record[name] = result[name]
    base_path.write_text(json.dumps(record, indent=2))
    np.savez_compressed(dst / IK, joint_names=result["joint_names"], joint_positions=result["joint_positions"], fps=np.float32(data_fps(result)))


def _solve(position, rotation, valid, group, fps=FPS, geometry=None, periodic=None):
    from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
    from curobo.kinematics import Kinematics, KinematicsCfg

    seed_r, seed_t = group.seed(position, valid)
    goal_p, goal_r = _hold(position, rotation, valid)
    offsets = _offset_grid(group.bounds)
    eligible = _eligible(offsets, position, valid, group, seed_r, seed_t, geometry)
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
    offset, arm = _best_seed(ik, goal_p, goal_r, seed_r, seed_t, offsets, group, eligible)
    offset, arm, losses = _refine(kin, goal_p, goal_r, valid, seed_r, seed_t, offset, arm, group,
                                fps=fps, geometry=geometry, periodic=periodic)
    return _camera_matrix(seed_r, seed_t, offset), arm, losses


# Synthetic rendering evaluates poses frame by frame, not actuator motion across stops.
# Only near-full-turn hinge ranges use a circular representation; other joints stay linear.
_PERIODIC_RANGE_GAP = np.deg2rad(1.0)


def _periodic_joints(robot, group):
    hinge = np.array([robot.model.joint(name).type[0] == mujoco.mjtJoint.mjJNT_HINGE for name in group.joints])
    return hinge & (group.upper - group.lower >= 2 * np.pi - _PERIODIC_RANGE_GAP)


def _joint_delta(a, b, periodic):
    delta = a - b
    if periodic is not None:
        delta = delta.copy()
        delta[..., periodic] = np.arctan2(np.sin(delta[..., periodic]), np.cos(delta[..., periodic]))
    return delta


def _lift_angles(arm, periodic):
    lifted = arm.copy()
    lifted[:, periodic] = np.unwrap(lifted[:, periodic], axis=0)
    return lifted


def _bounded_angles(q, lower, upper, periodic):
    center = (lower + upper) / 2
    wrapped = torch.remainder(q - center + np.pi, 2 * np.pi) - np.pi + center
    return torch.where(periodic, wrapped, q).clamp(lower, upper)


def _project_angles(q, lower, upper, periodic):
    bounded = _bounded_angles(q, lower, upper, periodic)
    # Optimize on a continuous angle lift, export only the bounded representation.
    turns = torch.round((q - bounded) / (2 * np.pi))
    return torch.where(periodic, bounded + turns * (2 * np.pi), bounded)


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


class NoBaseCandidate(ValueError):
    """The candidate grid has no base that meets the geometry constraints."""


def camera_size(src):
    _, height, width, _ = video_shape(src.with_suffix(".mp4"))
    return height, width


def base_boxes(robot):
    """Conservative bounds of visible geometry attached to each solve group's root."""
    model = robot.scene_spec().compile()
    robot.hide(model)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    boxes = {}
    for name, root in robot.roots.items():
        body = model.body(root).id
        points = []
        for i in np.flatnonzero((model.geom_bodyid == body) & (model.geom_rgba[:, 3] > 0)):
            if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[i]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
                world = vertices @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
                points.append((world - data.xpos[body]) @ data.xmat[body].reshape(3, 3))
            else:
                center = (data.geom_xpos[i] - data.xpos[body]) @ data.xmat[body].reshape(3, 3)
                radius = model.geom_rbound[i]
                points.append(np.array([center - radius, center + radius]))
        if points:
            xyz = np.concatenate(points)
            boxes[name] = np.array(list(product(*zip(xyz.min(0), xyz.max(0), strict=True))))
    return boxes


def candidate_mask(bases, position, valid, reach, box, intrinsic, size):
    """Reject near-camera, out-of-reach, or potentially visible bases."""
    center = bases[:, :3, 3]
    distance = np.linalg.norm(position[valid][None, :, :] - center[:, None, :], axis=-1)
    eligible = (np.linalg.norm(center, axis=-1) >= 0.20) & ((distance >= 0.08) & (distance <= 0.9 * reach)).all(1)
    if box is not None:
        height, width = size
        # Render derives a centered, square-pixel camera from fy and the image aspect ratio.
        f = float(intrinsic[1, 1])
        x, y, z = (box @ bases[:, :3, :3].transpose(0, 2, 1) + center[:, None]).transpose(2, 0, 1)
        outside = np.stack((z, width * z / 2 - f * x, width * z / 2 + f * x,
                            height * z / 2 - f * y, height * z / 2 + f * y))
        eligible &= (outside <= 0).all(axis=2).any(axis=0)
    return eligible


def _eligible(offsets, position, valid, group, seed_r, seed_t, geometry):
    if geometry is None:
        return np.ones(len(offsets), dtype=bool)
    box, intrinsic, size = geometry
    bases = np.stack([_camera_matrix(seed_r, seed_t, offset) for offset in offsets])
    eligible = candidate_mask(bases, position, valid, group.reach, box, intrinsic, size)
    if not eligible.any():
        raise NoBaseCandidate(f"{group.name}: no offscreen, reachable base among {len(offsets)} candidates")
    return eligible


def _candidate_choice(position_error, eligible):
    """Use the original full-trajectory mean position error among eligible bases."""
    score = np.nanmean(position_error, axis=1)
    score = np.where(eligible & np.isfinite(score), score, np.inf)
    if not np.isfinite(score).any():
        raise ValueError("no finite IK score for an eligible base")
    return int(score.argmin())


def _best_seed(ik, position, rotation, seed_r, seed_t, offsets, group, eligible):
    frames = len(position)
    posed = [
        _into_base(position, rotation, _camera_matrix(seed_r, seed_t, offset))
        for offset in offsets
    ]
    stacked_p = np.concatenate([item[0] for item in posed], axis=0)
    stacked_r = np.concatenate([item[1] for item in posed], axis=0)
    arm, error = _batch_ik(ik, stacked_p, stacked_r, group)
    arm = arm.reshape(len(offsets), frames, -1)
    error = error.reshape(len(offsets), frames)
    choice = _candidate_choice(error, eligible)
    return offsets[choice], arm[choice]


def _refine(kin, position, rotation, valid, seed_r, seed_t, offset, arm, group, fps=FPS, fixed_base=False, geometry=None, periodic=None):
    periodic = np.zeros(len(group.joints), dtype=bool) if periodic is None else np.asarray(periodic, dtype=bool)
    arm = _lift_angles(arm, periodic)
    pos, quat = _wxyz_pair(position, rotation)
    goal_p = torch.tensor(pos, device="cuda", dtype=torch.float32)
    goal_q = torch.tensor(quat, device="cuda", dtype=torch.float32)
    mask = torch.tensor(valid, device="cuda", dtype=torch.float32)
    seed_R = torch.tensor(seed_r, device="cuda", dtype=torch.float32)
    seed_t_gpu = torch.tensor(seed_t, device="cuda", dtype=torch.float32)
    lower, upper = _limits(group)
    q = torch.nn.Parameter(torch.tensor(arm, device="cuda", dtype=torch.float32))
    base = torch.nn.Parameter(torch.tensor(offset, device="cuda", dtype=torch.float32), requires_grad=not fixed_base)
    params = [{"params": [q], "lr": 2e-2}]
    if not fixed_base:
        params.append({"params": [base], "lr": 1e-2})
    opt = torch.optim.Adam(params, capturable=True, foreach=False)
    angle_mask = torch.as_tensor(periodic, device=q.device, dtype=torch.bool)
    with torch.no_grad():
        q.copy_(_project_angles(q, lower, upper, angle_mask))
    bounds = torch.tensor(group.bounds, device="cuda", dtype=torch.float32)
    def advance():
        opt.zero_grad(set_to_none=True)
        total, _terms = _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t_gpu, group,
                             (lower, upper), fps=fps, periodic=angle_mask)
        total.backward()
        torch.nn.utils.clip_grad_norm_([q, base], 5.0)
        opt.step()
        with torch.no_grad():
            q.copy_(_project_angles(q, lower, upper, angle_mask))
            base.clamp_(-bounds, bounds)

    # Warm optimizer/FK allocations outside capture. Each episode owns its graph.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(min(3, _STEPS)):
            advance()
    torch.cuda.current_stream().wait_stream(stream)
    if _STEPS > 3:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            advance()
        for _ in range(_STEPS - 3):
            graph.replay()
    if not (torch.isfinite(q).all() & torch.isfinite(base).all()):
        raise ValueError("non-finite trajectory refinement")
    refined_offset = offset if fixed_base else base.detach().cpu().numpy()
    if geometry is not None and not fixed_base:
        box, intrinsic, size = geometry
        matrix = _camera_matrix(seed_r, seed_t, refined_offset)
        if not candidate_mask(matrix[None], position, valid, group.reach, box, intrinsic, size)[0]:
            # Restart from the original IK seed at the eligible base. The optimized joints
            # belonged to a different base and cannot be reused as a finished trajectory.
            return _refine(kin, position, rotation, valid, seed_r, seed_t, offset, arm, group,
                           fps=fps, fixed_base=True, periodic=periodic)
    with torch.no_grad():
        total, terms = _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t_gpu, group, (lower, upper), fps=fps, periodic=angle_mask)
        smooth = _project_angles(torch.as_tensor(_smooth_arm(q.detach().cpu().numpy()), device=q.device),
                                 lower, upper, angle_mask)
        smooth_total, smooth_terms = _loss(
            kin, smooth, base, goal_p, goal_q, mask, seed_R, seed_t_gpu, group, (lower, upper), fps=fps, periodic=angle_mask,
        )
        if (smooth_total <= total and smooth_terms["position"] <= terms["position"] + 0.002
                and smooth_terms["orientation"] <= terms["orientation"] + 0.005):
            q, terms = smooth, smooth_terms
    losses = {key: float(value.detach().cpu()) for key, value in terms.items()}
    return refined_offset, _bounded_angles(q.detach(), lower, upper, angle_mask).cpu().numpy(), losses


def _loss(kin, q, base, goal_p, goal_q, mask, seed_R, seed_t, group, limits, fps=FPS, periodic=None):
    pos, quat = _fk_camera(kin, q, base, seed_R, seed_t, group)
    pos_err = torch.linalg.norm(pos - goal_p, dim=-1)
    dot = (quat * goal_q).sum(dim=-1).abs().clamp(0.0, 1.0)
    ori = 1.0 - dot.square()
    scale = mask.sum().clamp_min(1.0)
    position = (pos_err * mask).sum() / scale
    orientation = (ori * mask).sum() / scale
    delta = q[1:] - q[:-1]
    continuity = delta.square().mean() if len(delta) else q.sum() * 0.0
    velocity = (delta * valid_fps(fps)).square().mean() if len(delta) else q.sum() * 0.0
    jerk = (q[3:] - 3.0 * q[2:-1] + 3.0 * q[1:-2] - q[:-3]).square().mean() if len(q) >= 4 else q.sum() * 0.0
    low, high = limits
    bounded = q if periodic is None else _bounded_angles(q, low, high, periodic)
    joint_limit = torch.relu(low - bounded).square().mean() + torch.relu(bounded - high).square().mean()
    total = (_POSITION_WEIGHT * position + _ORIENTATION_WEIGHT * orientation + 4.0 * continuity
             + 0.05 * velocity + 2.0 * jerk + 5.0 * joint_limit)
    return total, {
        "position": position,
        "orientation": orientation,
        "continuity": continuity,
        "velocity": velocity,
        "jerk": jerk,
    }


class _RefinementFK(torch.autograd.Function):
    """Differentiate tool poses through the geometric Jacobian in the base frame.

    The bundled cuRobo quaternion backward projects a body-frame rotational
    covector onto base-frame joint axes. Ordinary PyTorch quaternion losses need
    a base-frame covector instead. Use the public geometric Jacobian, without
    patching cuRobo or depending on its quaternion-backward convention.
    """

    @staticmethod
    def forward(ctx, ordered, kin):
        from curobo.types import JointState

        previous = kin.compute_jacobian
        kin.compute_jacobian = True
        try:
            state = kin.compute_kinematics(JointState.from_position(ordered, joint_names=list(kin.joint_names)))
        finally:
            kin.compute_jacobian = previous
        # Kinematics reuses its buffers; saved values must survive another FK call.
        quaternion = state.tool_poses.quaternion.clone()
        ctx.save_for_backward(state.tool_jacobians.clone(), quaternion)
        return state.tool_poses.position.clone(), quaternion

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_position, grad_quaternion):
        jacobian, quaternion = ctx.saved_tensors
        if grad_position is None:
            grad_position = torch.zeros_like(quaternion[..., 1:])
        if grad_quaternion is None:
            angular = torch.zeros_like(grad_position)
        else:
            conjugate = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
            # q_dot = 0.5 * [0, omega_base] * q (wxyz quaternion order).
            angular = 0.5 * _quat_mul(grad_quaternion, conjugate)[..., 1:]
        covector = torch.cat((grad_position, angular), dim=-1)
        return torch.einsum("...lvd,...lv->...d", jacobian, covector), None


def _fk_camera(kin, q, base, seed_R, seed_t, group):
    from curobo._src.geom.transform import matrix_to_quaternion
    from curobo.types import JointState

    names = list(kin.joint_names)
    ordered = q if names == list(group.joints) else torch.stack([q[:, list(group.joints).index(name)] for name in names], dim=-1)
    ordered = ordered.reshape(ordered.shape[0], 1, ordered.shape[-1]).contiguous()
    if torch.is_grad_enabled() and ordered.requires_grad:
        position, quaternion = _RefinementFK.apply(ordered, kin)
        frames = list(kin.tool_frames)
    else:
        state = kin.compute_kinematics(JointState.from_position(ordered, joint_names=names))
        position, quaternion = state.tool_poses.position, state.tool_poses.quaternion
        frames = list(state.tool_poses.tool_frames)
    order = [frames.index(name) for name in group.tools]
    pos = torch.stack([position[:, 0, i] for i in order], dim=1).float()
    quat = torch.stack([quaternion[:, 0, i] for i in order], dim=1).float()
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
    return _arm_from_result(result, ik, group)[: len(pos)], result.position_error.detach().reshape(len(pos), -1).mean(dim=-1).cpu().numpy()


def _arm_from_result(result, ik, group) -> np.ndarray:
    state = result.js_solution
    values = state.position.detach().float().cpu().numpy()
    while values.ndim > 2:
        values = values[:, 0]
    names = list(state.joint_names or ik.joint_names)
    index = [names.index(name) for name in group.joints]
    return np.asarray(values[:, index], dtype=np.float64)


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
