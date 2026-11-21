"""cuRobo numerical backend for episode streams. No I/O or scheduling dependencies."""

from copy import copy

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares, minimize
from scipy.spatial.transform import Rotation

from egowhale.action import base_ik as core


def _continuous_solution(result, current, has_previous, names, periodic=None):
    """Prefer the nearest converged pose solution. Keep cuRobo's top rank as fallback."""
    result_names = list(result.js_solution.joint_names or names)
    joints = result.js_solution.position[..., [result_names.index(name) for name in names]]
    # cuRobo marks a seed successful only when feasible and within both pose tolerances.
    usable = result.success & torch.isfinite(joints).all(dim=-1)
    delta = joints - current[:, None, :]
    if periodic is not None:
        delta = torch.where(periodic, torch.atan2(torch.sin(delta), torch.cos(delta)), delta)
    distance = delta.square().sum(dim=-1)
    nearest = distance.masked_fill(~usable, torch.inf).argmin(dim=1)
    choice = torch.where(has_previous & usable.any(dim=1), nearest, 0)
    rows = torch.arange(len(choice), device=choice.device)
    return joints[rows, choice], result.position_error[rows, choice]


# Flag jumps at approximately 30 degrees per frame in the 18 FPS dataset.
_REPAIR_SPEED = 3.0 * np.pi
_REPAIR_REGIONS = 4


def _repair_jumps(robot, group, base, trajectory, position, rotation, valid, fps, branch_repair=None):
    """Retry short intervals with joint-step bounds and verify Cartesian tracking."""
    step = _REPAIR_SPEED / fps
    periodic = core._periodic_joints(robot, group)
    lower, upper = group.lower, group.upper
    distances = np.abs(core._joint_delta(trajectory[1:], trajectory[:-1], periodic)).max(axis=1)
    jumps = np.flatnonzero(distances > step + 1e-6)
    if not len(jumps):
        return trajectory, 0, 0, {}
    model = copy(robot.model)
    robot.place_bases(model, {group.name: base})
    data = mujoco.MjData(model)
    indices = [model.joint(name).qposadr[0] for name in group.joints]
    sites = [model.site(robot.tcp_sites[side]).id for side in group.sides]

    def errors(q, frame):
        data.qpos[indices] = q
        mujoco.mj_kinematics(model, data)
        active = valid[frame]
        p = data.site_xpos[sites][active] - position[frame, active]
        r = data.site_xmat[sites].reshape(-1, 3, 3)[active] @ rotation[frame, active].transpose(0, 2, 1)
        return p, Rotation.from_matrix(r).as_rotvec() if len(r) else np.empty((0, 3))

    changed = trajectory.astype(np.float64, copy=True)
    remaining = len(jumps)
    attempts = 0
    for edge in jumps[np.argsort(-distances[jumps], kind="stable")]:
        if np.max(np.abs(core._joint_delta(changed[edge + 1], changed[edge], periodic))) <= step + 1e-6:
            continue
        if attempts == _REPAIR_REGIONS:
            break
        attempts += 1
        for radius in (6, 18):
            start, stop = max(0, edge + 1 - radius), min(len(changed), edge + 1 + radius)
            a, b = max(0, start - 1), min(len(changed), stop + 1)
            old = changed[start:stop].copy()
            old_errors = [errors(trajectory[i], i) for i in range(start, stop)]
            old_jumps = (np.abs(core._joint_delta(changed[a + 1:b], changed[a:b - 1], periodic)) > step + 1e-6).any(axis=1).sum()
            for direction in (1, -1):
                anchor = start - 1 if direction == 1 else stop
                endpoint = stop if direction == 1 else start - 1
                frames = range(start, stop) if direction == 1 else range(stop - 1, start - 1, -1)
                previous = changed[anchor] if 0 <= anchor < len(changed) else changed[start if direction == 1 else stop - 1]
                candidate = old.copy()
                for frame in frames:
                    lo, hi = np.maximum(lower, previous - step), np.minimum(upper, previous + step)
                    if 0 <= endpoint < len(changed):
                        reach = step * abs(endpoint - frame)
                        lo = np.maximum(lo, changed[endpoint] - reach)
                        hi = np.minimum(hi, changed[endpoint] + reach)
                    if np.any(hi - lo <= 1e-8):
                        break

                    def residual(q, frame=frame, previous=previous):
                        p, r = errors(q, frame)
                        return np.r_[p.ravel() / 0.01, r.ravel() / 0.15, 0.05 * (q - previous)]

                    solution = least_squares(residual, np.clip(previous, lo, hi), bounds=(lo, hi), max_nfev=16)
                    p, r = errors(solution.x, frame)
                    old_p, old_r = old_errors[frame - start]
                    if (np.any(np.linalg.norm(p, axis=1) > np.maximum(0.010, np.linalg.norm(old_p, axis=1) + 0.002))
                            or np.any(np.linalg.norm(r, axis=1) > np.maximum(np.deg2rad(10), np.linalg.norm(old_r, axis=1) + np.deg2rad(10)))):
                        break
                    candidate[frame - start] = previous = solution.x
                else:
                    joined = np.concatenate((changed[a:start], candidate, changed[stop:b]))
                    count = (np.abs(core._joint_delta(joined[1:], joined[:-1], periodic)) > step + 1e-6).any(axis=1).sum()
                    if count < old_jumps:
                        changed[start:stop] = candidate
                        remaining += int(count - old_jumps)
                        break
            else:
                continue
            break
    changed = _repair_severe_jumps(robot, group, base, changed.astype(trajectory.dtype), trajectory,
                                  position, rotation, valid, fps)
    branch_terms = {}
    if branch_repair is not None:
        changed, branch_terms = branch_repair(changed, trajectory)
    remaining = int((np.abs(core._joint_delta(changed[1:], changed[:-1], periodic)).max(axis=1) > step + 1e-6).sum())
    terms = {}
    if not np.array_equal(changed, trajectory):
        position_errors, rotation_errors = [], []
        for frame, q in enumerate(changed):
            p, r = errors(q, frame)
            position_errors.extend(np.linalg.norm(p, axis=1))
            rotation_errors.extend(np.sin(np.linalg.norm(r, axis=1) / 2) ** 2)
        lifted = core._lift_angles(changed, periodic)
        delta = np.diff(lifted, axis=0)
        terms = dict(position=float(np.mean(position_errors)), orientation=float(np.mean(rotation_errors)),
                     continuity=float(np.square(delta).mean()), velocity=float(np.square(delta * fps).mean()),
                     jerk=float(np.square(np.diff(lifted, n=3, axis=0)).mean()) if len(changed) >= 4 else 0.0)
    terms.update(branch_terms)
    return changed.astype(trajectory.dtype), len(jumps), remaining, terms


# Additional local repair is bounded in work and in per-TCP pose degradation.
_CONSTRAINED_POSITION_SLACK = 0.002  # metres, relative to input of _repair_jumps
_CONSTRAINED_ROTATION_SLACK = np.deg2rad(10.0)
_CONSTRAINED_ITERATIONS = 120
_CONSTRAINED_SEVERE_STEP = np.pi / 2
_CONSTRAINED_FALLBACK_STEP = np.deg2rad(85.0)


def _rotation_log_jacobian(vector):
    """Derivative of Log(R) for a world-frame (left) angular perturbation."""
    theta = np.linalg.norm(vector)
    x, y, z = vector
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    coefficient = 1 / 12 if theta < 1e-5 else (1 - .5 * theta / np.tan(.5 * theta)) / theta**2
    return np.eye(3) - .5 * skew + coefficient * (skew @ skew)


def _constrained_interval(robot, group, base, trajectory, reference, position, rotation, valid,
                          start, stop, cap, linear_seed=False):
    """Solve a short interval; return None unless poses and both seams pass validation.

    reference is frozen across retries/overlapping intervals, so the 2 mm / 10 degree
    allowances cannot accumulate. Asset bases and targets remain unchanged.
    """
    periodic = core._periodic_joints(robot, group)
    # SLSQP and MuJoCo use float64; unwrap in that precision before solving.
    lifted = core._lift_angles(np.asarray(trajectory, dtype=np.float64), periodic)
    original = lifted[start:stop]
    count, dof = original.shape
    active = [(f, s) for f in range(count) for s in range(len(group.sides)) if valid[start + f, s]]
    if not active:
        return None
    model = copy(robot.model)
    robot.place_bases(model, {group.name: base})
    data = mujoco.MjData(model)
    indices = [model.joint(n).qposadr[0] for n in group.joints]
    dofs = [model.joint(n).dofadr[0] for n in group.joints]
    sites = [model.site(robot.tcp_sites[s]).id for s in group.sides]
    jp, jr = np.empty((3, model.nv)), np.empty((3, model.nv))
    prefix = lifted[start - 1:start] if start else np.empty((0, dof))
    suffix = lifted[stop:stop + 1]

    def poses(arm):
        p, r, pj, rj = [], [], [], []
        for frame in range(count):
            data.qpos[indices] = arm[frame]
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            for side, site in enumerate(sites):
                if not valid[start + frame, side]:
                    continue
                pv = data.site_xpos[site] - position[start + frame, side]
                rv = Rotation.from_matrix(
                    data.site_xmat[site].reshape(3, 3) @ rotation[start + frame, side].T,
                ).as_rotvec()
                mujoco.mj_jacSite(model, data, jp, jr, site)
                p.append(pv)
                r.append(rv)
                pj.append(jp[:, dofs].copy())
                rj.append(_rotation_log_jacobian(rv) @ jr[:, dofs])
        return np.asarray(p), np.asarray(r), np.asarray(pj), np.asarray(rj)

    ref_p, ref_r, _, _ = poses(reference[start:stop])
    # Leave a small numerical margin for SLSQP and export to float32.
    position_budget = np.linalg.norm(ref_p, axis=1) + _CONSTRAINED_POSITION_SLACK
    rotation_budget = np.linalg.norm(ref_r, axis=1) + _CONSTRAINED_ROTATION_SLACK
    pl, rl = position_budget - 1e-5, rotation_budget - .001
    cached = {}

    def evaluate(x):
        if "x" in cached and np.array_equal(x, cached["x"]):
            return cached["values"]
        arm = x.reshape(count, dof)
        pv, rv, pj, rj = poses(arm)
        joined = np.concatenate((prefix, arm, suffix))
        delta = np.diff(joined, axis=0)
        excess = np.sign(delta) * np.maximum(np.abs(delta) - cap * (25 / 60), 0.)
        objective = (np.sum(excess**2) + .01 * np.sum(delta**2)
                     + .0001 * np.sum((arm - original)**2) + .001 * np.sum(pv**2 / .005**2))
        edge_gradient = 2 * excess + .02 * delta
        joined_gradient = np.zeros_like(joined)
        joined_gradient[:-1] -= edge_gradient
        joined_gradient[1:] += edge_gradient
        gradient = joined_gradient[len(prefix):len(prefix) + count] + .0002 * (arm - original)
        constraint = np.r_[1 - np.sum(pv**2, axis=1) / pl**2, 1 - np.sum(rv**2, axis=1) / rl**2]
        jacobian = np.zeros((len(constraint), count * dof))
        for k, (frame, _) in enumerate(active):
            columns = slice(frame * dof, (frame + 1) * dof)
            jacobian[k, columns] = -2 * pv[k] @ pj[k] / pl[k]**2
            jacobian[len(active) + k, columns] = -2 * rv[k] @ rj[k] / rl[k]**2
            gradient[frame] += .002 * pv[k] @ pj[k] / .005**2
        values = objective, gradient.ravel(), constraint, jacobian, pv, rv
        cached.update(x=x.copy(), values=values)
        return values

    # Linear differences include the fixed adjacent poses outside the repaired interval.
    mapping = np.zeros((count + len(prefix) + len(suffix), count))
    mapping[len(prefix):len(prefix) + count] = np.eye(count)
    matrix = np.kron(np.diff(mapping, axis=0), np.eye(dof))
    constant = np.diff(np.concatenate((prefix, np.zeros_like(original), suffix)), axis=0).ravel()
    lower, upper = group.lower.copy(), group.upper.copy()
    lower[periodic], upper[periodic] = -np.inf, np.inf
    initial = original.copy()
    if linear_seed and len(prefix) and len(suffix):
        initial = np.linspace(prefix[0], suffix[0], count + 2)[1:-1]
    best = None

    def retain_feasible(x):
        nonlocal best
        values = evaluate(x)
        if (np.isfinite(x).all()
                and np.all(np.linalg.norm(values[4], axis=1) <= position_budget)
                and np.all(np.linalg.norm(values[5], axis=1) <= rotation_budget)
                and np.max(np.abs(matrix @ x + constant), initial=0.) <= cap + 1e-7):
            if best is None or values[0] < best[0]:
                best = values[0], x.copy()

    result = minimize(
        lambda x: evaluate(x)[0], np.clip(initial, lower, upper).ravel(),
        jac=lambda x: evaluate(x)[1], method="SLSQP",
        bounds=list(zip(np.tile(lower, count), np.tile(upper, count), strict=True)),
        constraints=[
            dict(type="ineq", fun=lambda x: np.r_[cap - (matrix @ x + constant), cap + (matrix @ x + constant)],
                 jac=lambda x: np.r_[-matrix, matrix]),
            dict(type="ineq", fun=lambda x: evaluate(x)[2], jac=lambda x: evaluate(x)[3]),
        ],
        callback=retain_feasible, options=dict(maxiter=_CONSTRAINED_ITERATIONS, ftol=1e-8),
    )
    if not np.isfinite(result.x).all():
        return None
    retain_feasible(result.x)
    # A last rejected or iteration-limited step must not discard an earlier valid path.
    solution = result.x if best is None else best[1]
    arm = solution.reshape(count, dof).copy()
    center = (group.lower + group.upper) / 2
    arm[:, periodic] = (arm[:, periodic] - center[periodic] + np.pi) % (2 * np.pi) - np.pi + center[periodic]
    arm = np.clip(arm, group.lower, group.upper).astype(trajectory.dtype)
    pv, rv, _, _ = poses(arm)
    joined = np.concatenate((prefix, arm, suffix))
    steps = core._joint_delta(joined[1:], joined[:-1], periodic)
    # Solver success alone is not evidence of a valid result; check actual exported FK.
    if (np.any(np.linalg.norm(pv, axis=1) > position_budget + 1e-7)
            or np.any(np.linalg.norm(rv, axis=1) > rotation_budget + 1e-7)
            or np.max(np.abs(steps), initial=0.) > cap + 1e-6):
        return None
    return arm


def _repair_severe_jumps(robot, group, base, trajectory, reference, position, rotation, valid, fps):
    """Bounded local retries, no inference scheduling or file I/O dependencies."""
    periodic = core._periodic_joints(robot, group)
    step = _REPAIR_SPEED / fps
    changed = trajectory.copy()

    def severity(arm):
        delta = np.abs(core._joint_delta(arm[1:], arm[:-1], periodic)).max(axis=1)
        return (int((delta > _CONSTRAINED_SEVERE_STEP + 1e-6).sum()),
                int((delta > step + 1e-6).sum()), float(delta.max(initial=0.)))

    delta = np.abs(core._joint_delta(changed[1:], changed[:-1], periodic)).max(axis=1)
    edges = np.flatnonzero(delta > _CONSTRAINED_SEVERE_STEP + 1e-6)
    for edge in edges[np.argsort(-delta[edges], kind="stable")][:_REPAIR_REGIONS]:
        if np.max(np.abs(core._joint_delta(changed[edge + 1], changed[edge], periodic))) <= _CONSTRAINED_SEVERE_STEP + 1e-6:
            continue
        before = severity(changed)
        short, long = max(2, round(fps / 3)), max(3, round(fps * 2 / 3))
        cap = min(2 * step, np.deg2rad(60))
        attempts = [(radius, cap, linear) for radius in (short, long) for linear in (False, True)]
        # One final retry with a less restrictive joint-step cap; pose budgets stay fixed.
        attempts.append((long, max(cap, min(3 * step, _CONSTRAINED_FALLBACK_STEP)), False))
        for radius, cap, linear_seed in attempts:
            start, stop = max(0, edge + 1 - radius), min(len(changed), edge + 1 + radius)
            arm = _constrained_interval(robot, group, base, changed, reference, position, rotation, valid,
                                        start, stop, cap, linear_seed)
            if arm is None:
                continue
            candidate = changed.copy()
            candidate[start:stop] = arm
            if severity(candidate) < before:
                changed = candidate
                break
    return changed


# Local branch search uses the existing frozen per-TCP repair budget.
_BRANCH_TRIGGER = np.deg2rad(45.)
_BRANCH_MIN_GAIN = np.deg2rad(5.)
_BRANCH_REGIONS = 4
_BRANCH_SEEDS = 8
_BRANCH_BATCH = 256
_BRANCH_JITTER = np.deg2rad(9.)


def _branch_windows(trajectory, periodic, fps):
    distance = np.abs(core._joint_delta(trajectory[1:], trajectory[:-1], periodic)).max(axis=1)
    radius = max(2, round(fps * 2 / 3))
    windows = []
    for edge in np.argsort(-distance, kind="stable"):
        if distance[edge] <= _BRANCH_TRIGGER:
            break
        if any(lo <= edge and edge + 1 < hi for lo, hi in windows):
            continue
        windows.append((max(0, int(edge) + 1 - radius), min(len(trajectory), int(edge) + 1 + radius)))
        if len(windows) == _BRANCH_REGIONS:
            break
    # Keep each solver call small. Overlaps use the same frozen reference budget.
    return windows


def _branch_path(pool, pe, re, reference_pe, reference_re, valid, periodic, lower, upper,
                 fixed_first=False, fixed_last=False, step=np.pi / 6):
    """Minimax selection in a finite candidate graph; original path is column zero."""
    finite = np.isfinite(pool).all(-1) & np.isfinite(pe).all(-1) & np.isfinite(re).all(-1)
    usable = finite & ((pool >= lower - 1e-6) & (pool <= upper + 1e-6)).all(-1)
    usable &= (((pe <= reference_pe[:, None] + _CONSTRAINED_POSITION_SLACK - 1e-5)
                & (re <= reference_re[:, None] + _CONSTRAINED_ROTATION_SLACK - .001))
               | ~valid[:, None]).all(-1)
    # Column zero is the already validated input, never a new use of the allowance.
    usable[:, 0] = True
    if fixed_first:
        usable[0, 1:] = False
    if fixed_last:
        usable[-1, 1:] = False
    indices = []
    for frame in range(len(pool)):
        ids = np.flatnonzero(usable[frame])
        _, unique = np.unique(np.round(pool[frame, ids], 4), axis=0, return_index=True)
        indices.append(ids[np.sort(unique)])
    peak = np.zeros(len(indices[0]))
    for frame in range(1, len(pool)):
        delta = core._joint_delta(pool[frame - 1, indices[frame - 1]][:, None],
                                  pool[frame, indices[frame]][None], periodic)
        peak = np.maximum(peak[:, None], np.abs(delta).max(-1)).min(0)
    cap = float(peak.min()) + 1e-6
    original_peak = np.abs(core._joint_delta(pool[1:, 0], pool[:-1, 0], periodic)).max(initial=0.)
    if cap + _BRANCH_MIN_GAIN >= original_peak:
        return pool[:, 0].copy()

    def pose_cost(frame):
        ids = indices[frame]
        return ((100 * pe[frame, ids]**2 + .1 * re[frame, ids]**2) * valid[frame]).sum(-1)

    cost, parents = pose_cost(0), []
    for frame in range(1, len(pool)):
        delta = core._joint_delta(pool[frame - 1, indices[frame - 1]][:, None],
                                  pool[frame, indices[frame]][None], periodic)
        edges = (cost[:, None] + 100 * np.maximum(np.abs(delta) - step, 0.).sum(-1)**2
                 + np.square(delta).sum(-1))
        edges[np.abs(delta).max(-1) > cap] = np.inf
        parent = edges.argmin(0)
        cost = edges[parent, np.arange(len(parent))] + pose_cost(frame)
        parents.append(parent)
    selected = [int(cost.argmin())]
    for parent in reversed(parents):
        selected.append(int(parent[selected[-1]]))
    picks = [indices[f][i] for f, i in enumerate(reversed(selected))]
    return pool[np.arange(len(pool)), picks].copy()


class _BranchRepair:
    """Asset-driven, bounded IK candidate search; independent of node scheduling/I/O."""

    def __init__(self, canonical, kin):
        self.canonical, self.kin = canonical, kin
        self.solver = None

    def _solver(self):
        if self.solver is None:
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
            self.solver = InverseKinematics(InverseKinematicsCfg.create(
                robot=self.canonical.config, num_seeds=_BRANCH_SEEDS, seed_solver_num_seeds=32,
                self_collision_check=False, load_collision_spheres=False, use_cuda_graph=True,
                max_batch_size=_BRANCH_BATCH, optimizer_configs=["ik/lbfgs_ik.yml"],
            ))
            if self.solver.seed_ik_solver is not None:
                self.solver.seed_ik_solver.config.use_cuda_graph = False
        return self.solver

    def _poses(self, q, base):
        def tensor(x):
            return torch.as_tensor(x, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            return core._fk_camera(self.kin, tensor(q), torch.zeros(4, device="cuda"),
                                   tensor(base[:3, :3]), tensor(base[:3, 3]), self.canonical)

    def _pool(self, trajectory, base, position, rotation, periodic, lower, upper):
        from curobo.types import GoalToolPose, JointState, Pose
        actual_p, actual_q = self._poses(trajectory, base)
        actual_p, actual_q = actual_p.cpu().numpy(), actual_q.cpu().numpy()
        actual_r = Rotation.from_quat(actual_q.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix()
        actual_r = actual_r.reshape(rotation.shape)
        delta = Rotation.from_matrix((rotation @ actual_r.transpose(0, 1, 3, 2)).reshape(-1, 3, 3)).as_rotvec()
        delta = delta.reshape(*rotation.shape[:2], 3)
        fractions = (0., .5, 1.)
        centers_r = np.stack([
            Rotation.from_rotvec((delta * f).reshape(-1, 3)).as_matrix().reshape(rotation.shape) @ actual_r
            for f in fractions
        ], axis=1)
        jitter = Rotation.from_rotvec(np.r_[np.zeros((1, 3)), np.eye(3) * _BRANCH_JITTER,
                                            -np.eye(3) * _BRANCH_JITTER]).as_matrix()
        orientations = np.einsum("kij,fasjl->faksil", jitter, centers_r).reshape(
            len(trajectory), 21, len(self.canonical.tools), 3, 3)
        goals_r = np.repeat(orientations[:, None], 3, axis=1).reshape(-1, len(self.canonical.tools), 3, 3)
        centers_p = np.stack([actual_p * (1-f) + position * f for f in fractions], axis=1)
        goals_p = np.repeat(centers_p[:, :, None], 21, axis=2).reshape(-1, len(self.canonical.tools), 3)
        p, r = core._into_base(goals_p, goals_r, base)
        p, quat = core._wxyz_pair(p, r)
        seeds = np.repeat(trajectory, 63, axis=0)
        solver = self._solver()
        names = list(solver.joint_names)
        order = [list(self.canonical.joints).index(n) for n in names]
        values = []
        for first in range(0, len(p), _BRANCH_BATCH):
            last = min(first + _BRANCH_BATCH, len(p))
            pad = _BRANCH_BATCH - (last-first)
            pp = np.pad(p[first:last], ((0, pad), (0, 0), (0, 0)), mode="edge")
            qq = np.pad(quat[first:last], ((0, pad), (0, 0), (0, 0)), mode="edge")
            cc = np.pad(seeds[first:last, order], ((0, pad), (0, 0)), mode="edge")
            goal = GoalToolPose.from_poses({
                n: Pose(position=torch.as_tensor(pp[:, i], device="cuda", dtype=torch.float32),
                        quaternion=torch.as_tensor(qq[:, i], device="cuda", dtype=torch.float32),
                        name=n, normalize_rotation=True) for i, n in enumerate(self.canonical.tools)
            }, ordered_tool_frames=list(self.canonical.tools))
            current = JointState.from_position(torch.as_tensor(cc, device="cuda", dtype=torch.float32).contiguous(),
                                               joint_names=names)
            result = solver.solve_pose(goal_tool_poses=goal, current_state=current, return_seeds=_BRANCH_SEEDS)
            result_order = [list(result.js_solution.joint_names).index(n) for n in self.canonical.joints]
            values.append(result.js_solution.position[:last-first, :, result_order].cpu().numpy())
        pool = np.concatenate(values).reshape(len(trajectory), 63 * _BRANCH_SEEDS, -1)
        center = (lower + upper) / 2
        pool[:, :, periodic] = (pool[:, :, periodic] - center[periodic] + np.pi) % (2*np.pi) - np.pi + center[periodic]
        pool = np.clip(pool, lower, upper).astype(trajectory.dtype)
        return np.concatenate((trajectory[:, None], pool), axis=1)

    def _errors(self, pool, base, position, rotation):
        _, goal_q = core._wxyz_pair(position, rotation)
        flat = pool.reshape(-1, pool.shape[-1])
        pe, re = [], []
        for first in range(0, len(flat), 4096):
            last = min(first + 4096, len(flat))
            frame = np.arange(first, last) // pool.shape[1]
            pp, qq = self._poses(flat[first:last], base)
            goal = torch.as_tensor(goal_q[frame], device="cuda", dtype=torch.float32)
            goal = goal / goal.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            conjugate = goal.clone()
            conjugate[..., 1:] *= -1
            relative = core._quat_mul(qq, conjugate)
            pe.append(torch.linalg.norm(pp - torch.as_tensor(position[frame], device="cuda", dtype=torch.float32), dim=-1).cpu().numpy())
            re.append((2 * torch.atan2(relative[..., 1:].norm(dim=-1), relative[..., 0].abs())).cpu().numpy())
        shape = (*pool.shape[:2], len(self.canonical.tools))
        return np.concatenate(pe).reshape(shape), np.concatenate(re).reshape(shape)

    def __call__(self, robot, group, base, trajectory, reference, position, rotation, valid, fps):
        periodic = core._periodic_joints(robot, group)
        windows = _branch_windows(trajectory, periodic, fps)
        changed = trajectory.copy()
        accepted = 0
        for lo, hi in windows:
            a, b = max(0, lo-1), min(len(changed), hi+1)
            pool = self._pool(changed[a:b], base, position[a:b], rotation[a:b],
                              periodic, group.lower, group.upper)
            # Recompute every pose after limit projection, including every TCP.
            pe, re = self._errors(pool, base, position[a:b], rotation[a:b])
            rp, rr = self._errors(reference[a:b, None], base, position[a:b], rotation[a:b])
            path = _branch_path(pool, pe, re, rp[:, 0], rr[:, 0], valid[a:b], periodic,
                                group.lower, group.upper, a < lo, b > hi, _REPAIR_SPEED/fps)
            if np.array_equal(path, changed[a:b]):
                continue
            # Independent CPU FK validates the exported values, not just solver status.
            if not _valid_branch(robot, group, base, path, changed[a:b], reference[a:b],
                                 position[a:b], rotation[a:b], valid[a:b]):
                continue
            changed[a:b] = path
            accepted += 1
        return changed, dict(branch_windows=len(windows), branch_windows_accepted=accepted)


def _valid_branch(robot, group, base, path, previous, reference, position, rotation, valid):
    if not np.isfinite(path).all() or not ((path >= group.lower-1e-6) & (path <= group.upper+1e-6)).all():
        return False
    model = copy(robot.model)
    robot.place_bases(model, {group.name: base})
    data = mujoco.MjData(model)
    indices = [model.joint(n).qposadr[0] for n in group.joints]
    sites = [model.site(robot.tcp_sites[s]).id for s in group.sides]
    for f in np.flatnonzero(np.any(path != previous, axis=1)):
        errors = []
        for q in (reference[f], path[f]):
            data.qpos[indices] = q
            mujoco.mj_kinematics(model, data)
            pv = np.linalg.norm(data.site_xpos[sites] - position[f], axis=-1)
            rv = Rotation.from_matrix(data.site_xmat[sites].reshape(-1, 3, 3) @ rotation[f].transpose(0, 2, 1)).magnitude()
            errors.append((pv, rv))
        active = valid[f]
        if (np.any(errors[1][0][active] > errors[0][0][active] + _CONSTRAINED_POSITION_SLACK)
                or np.any(errors[1][1][active] > errors[0][1][active] + _CONSTRAINED_ROTATION_SLACK)):
            return False
    return True


class CuroboBackend:
    def __init__(self, robot, capacity):
        self.robot = robot
        self.groups = list(robot.groups)
        self.keys = [g.batch_key or ("group", i) for i, g in enumerate(self.groups)]
        self.buckets = {}
        for key, g in zip(self.keys, self.groups, strict=True):
            self.buckets.setdefault(key, []).append(g)
        for groups in self.buckets.values():
            g = groups[0]
            for other in groups[1:]:
                if len(g.joints) != len(other.joints) or len(g.tools) != len(other.tools):
                    raise ValueError("incompatible asset batch_key")
                np.testing.assert_allclose(g.lower, other.lower)
                np.testing.assert_allclose(g.upper, other.upper)
        self.batch_widths = {
            key: capacity * sum(len(core._offset_grid(group.bounds)) for group in groups)
            for key, groups in self.buckets.items()
        }
        self.solvers, self.kinematics, self.branch_repairs = {}, {}, {}
        self.base_bounds = None

    def reset(self):
        self.solvers.clear()
        self.kinematics.clear()
        self.branch_repairs.clear()
        import gc

        gc.collect()

    def base_candidates(self, offsets, position, valid, group, seed_r, seed_t, geometry):
        eligible = core._eligible(offsets, position, valid, group, seed_r, seed_t, geometry)
        return eligible, geometry

    def prepare(self, data):
        core.data_fps(data)  # Validate before allocating any GPU state.
        length = len(data["position"])
        if length < 1:
            raise ValueError("empty episode")
        tasks = []
        if "camera_size" in data and self.base_bounds is None:
            self.base_bounds = core.base_boxes(self.robot)
        for group, key in zip(self.groups, self.keys, strict=True):
            sides = list(group.sides)
            valid = data["valid"][:, sides]
            position, rotation = data["position"][:, sides], data["rotation"][:, sides]
            seed_r, seed_t = group.seed(position, valid)
            gp, gr = core._hold(position, rotation, valid)
            offsets = core._offset_grid(group.bounds)
            geometry = None if "camera_size" not in data else (
                self.base_bounds.get(group.name), data["intrinsic"], data["camera_size"],
            )
            eligible, geometry = self.base_candidates(
                offsets, position, valid, group, seed_r, seed_t, geometry,
            )
            posed = [core._into_base(gp, gr, core._camera_matrix(seed_r, seed_t, o)) for o in offsets]
            p, q = core._wxyz_pair(np.stack([p for p, r in posed], axis=1), np.stack([r for p, r in posed], axis=1))
            tasks.append(
                dict(
                    key=key,
                    group=group,
                    valid=valid,
                    gp=gp,
                    gr=gr,
                    seed_r=seed_r,
                    seed_t=seed_t,
                    offsets=offsets,
                    eligible=eligible,
                    geometry=geometry,
                    p=torch.as_tensor(p, device="cuda", dtype=torch.float32),
                    q=torch.as_tensor(q, device="cuda", dtype=torch.float32),
                    current=None,
                    arms=[],
                    errors=[],
                )
            )
        return dict(data=data, length=length, frame=0, tasks=tasks)

    def solver(self, key, width=None):
        width = self.batch_widths[key] if width is None else width
        cache_key = (key, width)
        if cache_key not in self.solvers:
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

            groups = self.buckets[key]
            self.solvers[cache_key] = InverseKinematics(
                InverseKinematicsCfg.create(
                    robot=groups[0].config,
                    num_seeds=core._SEEDS,
                    self_collision_check=False,
                    load_collision_spheres=False,
                    use_cuda_graph=True,
                    max_batch_size=width,
                    optimizer_configs=["ik/lbfgs_ik.yml"],
                )
            )
            # This cuRobo revision's LM graph faults on the second streamed call.
            # Keep LM eager while capturing the stable-width L-BFGS optimizer.
            if self.solvers[cache_key].seed_ik_solver is not None:
                self.solvers[cache_key].seed_ik_solver.config.use_cuda_graph = False
        return self.solvers[cache_key]

    def step(self, states):
        from curobo.types import GoalToolPose, JointState, Pose

        for key, groups in self.buckets.items():
            group = groups[0]
            tasks = [(s, t) for s in states for t in s["tasks"] if t["key"] == key]
            # Separate cached solvers keep their own fixed CUDA-graph shapes.
            # Tail episodes need not solve dummy rows for the initial full capacity.
            slots = 1 << (len(states) - 1).bit_length()
            rows_per_slot = sum(len(core._offset_grid(g.bounds)) for g in groups)
            width = min(self.batch_widths[key], slots * rows_per_slot)
            ik = self.solver(key, width)
            names = list(ik.joint_names)
            order = [list(group.joints).index(n) for n in names]
            p = torch.cat([t["p"][s["frame"]] for s, t in tasks])
            q = torch.cat([t["q"][s["frame"]] for s, t in tasks])
            currents = []
            previous = []
            for s, t in tasks:
                previous.extend([t["current"] is not None] * len(t["offsets"]))
                if t["current"] is None:
                    # New episode never inherits the removed episode's state, even when sharing its slot.
                    home = torch.as_tensor(t["group"].home, device=p.device, dtype=p.dtype)
                    currents.append(home[order].repeat(len(t["offsets"]), 1))
                else:
                    currents.append(t["current"])
            current = torch.cat(currents)
            count = len(p)
            padding = width - count
            if padding < 0:
                raise ValueError(f"IK batch for {key} has {count} rows, capacity is {width}")
            if padding:
                # Keep cuRobo's cached batch shape fixed even as episodes finish at different times.
                # A matched pose and current state from a real row give the dummy rows valid inputs.
                p = torch.cat((p, p[:1].expand(padding, -1, -1)))
                q = torch.cat((q, q[:1].expand(padding, -1, -1)))
                current = torch.cat((current, current[:1].expand(padding, -1)))
            previous.extend([False] * padding)
            has_previous = torch.as_tensor(previous, device=p.device, dtype=torch.bool)
            current_state = JointState.from_position(current, joint_names=names)
            goals = GoalToolPose.from_poses(
                {
                    name: Pose(position=p[:, side], quaternion=q[:, side], name=name, normalize_rotation=True)
                    for side, name in enumerate(group.tools)
                },
                ordered_tool_frames=list(group.tools),
            )
            result = ik.solve_pose(goal_tool_poses=goals, current_state=current_state, return_seeds=core._SEEDS)
            periodic = torch.as_tensor(core._periodic_joints(self.robot, group)[order], device=p.device)
            values, error = _continuous_solution(result, current, has_previous, names, periodic)
            # Only real rows update state or contribute to base selection.
            values, error = values[:count], error[:count]
            start = 0
            for s, t in tasks:
                stop = start + len(t["offsets"])
                t["current"] = values[start:stop].clone()
                t["arms"].append(values[start:stop, [names.index(n) for n in group.joints]].clone())
                t["errors"].append(error[start:stop].clone())
                start = stop
        for s in states:
            s["frame"] += 1

    def finish(self, state):
        from curobo.kinematics import Kinematics, KinematicsCfg

        arms, bases, losses = [], {}, {}
        for task in state["tasks"]:
            key = task["key"]
            canonical = self.buckets[key][0]
            if key not in self.kinematics:
                self.kinematics[key] = Kinematics(
                    KinematicsCfg.from_robot_yaml_file(canonical.config, tool_frames=list(canonical.tools))
                )
            candidates = torch.stack(task["arms"]).detach().float()
            error = torch.stack(task["errors"]).detach().float()
            if not (torch.isfinite(candidates).all() & torch.isfinite(error).all()):
                raise ValueError("non-finite sequential IK result")
            eligible = torch.as_tensor(task["eligible"], device=error.device)
            choice = int(error.mean(dim=0).masked_fill(~eligible, torch.inf).argmin())
            arm = candidates[:, choice].cpu().numpy()
            offset, trajectory, terms = core._refine(
                self.kinematics[key],
                task["gp"],
                task["gr"],
                task["valid"],
                task["seed_r"],
                task["seed_t"],
                task["offsets"][choice],
                arm,
                canonical,
                fps=core.data_fps(state["data"]),
                geometry=task["geometry"],
                periodic=core._periodic_joints(self.robot, task["group"]),
            )
            group = task["group"]
            base = core._camera_matrix(task["seed_r"], task["seed_t"], offset)
            if key not in self.branch_repairs:
                self.branch_repairs[key] = _BranchRepair(canonical, self.kinematics[key])

            def repair_branches(current, reference, key=key, group=group, base=base, task=task):
                return self.branch_repairs[key](
                    self.robot, group, base, current, reference, task["gp"], task["gr"], task["valid"],
                    core.data_fps(state["data"]),
                )

            trajectory, jumps_before, jumps_after, repaired_terms = _repair_jumps(
                self.robot, group, base, trajectory, task["gp"], task["gr"], task["valid"],
                core.data_fps(state["data"]), branch_repair=repair_branches,
            )
            arms.append(trajectory)
            bases[group.name] = base
            # Geometry-aware steps and raw coordinate cuts are separate diagnostics.
            raw_steps = np.abs(np.diff(trajectory, axis=0)).max(axis=1)
            raw_cuts = int((raw_steps > _REPAIR_SPEED / core.data_fps(state["data"]) + 1e-6).sum())
            repaired_terms.setdefault("branch_windows", 0)
            repaired_terms.setdefault("branch_windows_accepted", 0)
            terms.update(repaired_terms, jump_edges_before=jumps_before, jump_edges_after=jumps_after,
                         raw_coordinate_jump_edges=raw_cuts)
            losses[task["group"].name] = terms
        data = state["data"]
        return {
            **self.robot.pack(arms, data["width"], data["valid"]),
            "fps": core.data_fps(data),
            "bases": bases,
            "group_losses": losses,
            "losses": {k: float(np.mean([v[k] for v in losses.values()])) for k in next(iter(losses.values()))},
        }
