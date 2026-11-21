"""Camera-fixed base search and sequential GPU IK, independent of episode transport."""
from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F

from egowhale.media import data_fps

@dataclass(frozen=True)
class SearchOptions:
    nx: int = 5
    ny: int = 3
    nz: int = 3
    top: int = 5
    keys: int = 20
    search_steps: int = 12
    track_steps: int = 8
    orientation: float = .05
    track_step: float = .38
    placement: float = .045
    upright: bool = True
    bidirectional: bool = True
    motion_penalty: float = .0015
    frustum: float = .08
    tail_mix: float = .55
    tracking: str = "rated"


    @property
    def candidates(self):
        return 3 * self.nx * self.ny * self.nz

    def episode_cost(self, groups):
        return groups * max(self.candidates, self.top**groups if groups > 1 else 1)


WARMUP_STEPS = 20
DAMPING = .035
SEARCH_JOINT_STEP = .40
# A dataset motion ceiling, additionally limited by each asset's kinematic config.
JOINT_SPEED_CEILING = 5.5
ACCELERATION_WEIGHT = .0025


def _gpu(value):
    return torch.as_tensor(value, dtype=torch.float32, device="cuda")


def _keyframes(position, rotation, valid, keyframes):
    """Start/middle/end and spatial extremes per hand; fill with motion samples."""
    count = len(position)
    rows = torch.arange(count, device=position.device)
    priority = torch.zeros(count, dtype=torch.long, device=position.device)
    for side in range(position.shape[1]):
        v, p, r = valid[:, side], position[:, side], rotation[:, side]
        cumulative = v.long().cumsum(0)
        total = cumulative[-1]
        extremes = torch.cat((
            (cumulative >= 1).long().argmax().reshape(1),
            (cumulative > total // 2).long().argmax().reshape(1),
            (v.long() * (rows + 1)).argmax().reshape(1),
            p.masked_fill(~v[:, None], torch.inf).argmin(dim=0),
            p.masked_fill(~v[:, None], -torch.inf).argmax(dim=0),
        ))
        priority.scatter_reduce_(0, extremes, (total > 0).long().expand_as(extremes) * 2,
                                 reduce="amax")
        # Accumulate displacement/rotation between successive valid frames and sample it.
        last = torch.where(v, rows, 0).cummax(0).values
        prev = torch.cat((last[:1], last[:-1]))
        dp = torch.linalg.vector_norm(p - p[prev], dim=-1) / .05
        cosine = ((r * r[prev]).sum((-2, -1)) - 1) / 2
        dr = cosine.clamp(-1, 1).acos() / .35
        movement = torch.where(v & (cumulative > 1), torch.maximum(dp, dr), 0).cumsum(0)
        query = torch.linspace(0, 1, keyframes + 2, device=p.device)[1:-1] * movement[-1]
        samples = torch.searchsorted(movement.contiguous(), query.contiguous()).clamp(max=count - 1)
        priority.scatter_reduce_(0, samples, (total > 0).long().expand_as(samples), reduce="amax")
    # Spread ties over time, and fill unused slots by repeating the last selected frame.
    temporal = ((rows * 17) % 23) / 23
    raw = (priority * 2 + temporal).topk(min(count, keyframes)).indices
    selected = priority[raw] > 0
    latest = torch.where(selected, raw, 0).max()
    raw = torch.where(selected, raw, latest).sort().values
    return torch.cat((raw, latest.expand(keyframes - len(raw))))


def _rotation_error(goal, actual):
    """Shortest-arc orientation residual in the base frame (wxyz)."""
    gw, gv = goal[..., :1], goal[..., 1:]
    aw, av = actual[..., :1], actual[..., 1:]
    scalar = gw * aw + (gv * av).sum(-1, keepdim=True)
    vector = aw * gv - gw * av - torch.linalg.cross(gv, av)
    vector = torch.where(scalar < 0, -vector, vector)
    sine = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector * (2 * torch.atan2(sine, scalar.abs()) / sine.clamp_min(1e-7))


def _candidates(state, group, side_sign, is_shared, options, box=None):
    p, valid = state["p"][:, group.sides], state["valid"][:, group.sides]
    count = valid.sum((0, 1))
    c = (p * valid[..., None]).sum((0, 1)) / count.clamp_min(1)
    # Targets and candidates are in the source camera frame for the whole clip.
    swap = state["swap"]
    sign = torch.where(swap, -side_sign, side_sign)

    def offsets(lo, hi, count):
        if count == 1:
            return _gpu([(lo + hi) / 2])
        return torch.linspace(lo, hi, count, device=p.device)

    x = offsets(*((-.35, .35) if is_shared else (.32, .82)), options.nx)
    y = offsets(.1, .60, options.ny) + swap * -.15
    z = offsets(-.35, .15, options.nz) + swap * .45
    offset = torch.cartesian_prod(x, y, z)
    offset[:, 0] *= sign
    offset *= group.reach
    translation = c + offset
    away = _gpu((side_sign * 1.5, .6, -.7))
    translation = torch.where(count > 0, translation, away)
    # Same centroid-facing construction as the reference. Three pitch/yaw/roll
    # profiles per location approximate the larger upstream orientation grid.
    profiles = ((0., -20., 0.), (15., 0., 0.), (30., 20., 0.)) if options.upright else (
        (30., -20., -10.), (45., 0., 0.), (60., 20., 10.))
    tilt, yaw, roll = _gpu(profiles).deg2rad().unbind(-1)
    yaw, roll = yaw * sign, roll * sign
    up = torch.stack((torch.zeros_like(tilt), -tilt.cos(), -tilt.sin()), -1)
    view = -offset[:, None]  # Look towards the centroid, in the initial camera frame.
    forward = view - (view * up).sum(-1, keepdim=True) * up
    forward = torch.where(torch.linalg.vector_norm(forward, dim=-1, keepdim=True) < 1e-4,
                          _gpu((1., 0., 0.)), forward)
    forward = F.normalize(forward, dim=-1)
    lateral = F.normalize(torch.linalg.cross(up.expand_as(forward), forward), dim=-1)
    # Right-multiply yaw about the base z axis, then roll about the original forward axis.
    f1 = forward * yaw.cos()[None, :, None] + lateral * yaw.sin()[None, :, None]
    l1 = lateral * yaw.cos()[None, :, None] - forward * yaw.sin()[None, :, None]
    def rotate(v):
        return (v * roll.cos()[None, :, None] + torch.linalg.cross(forward, v) * roll.sin()[None, :, None]
                + forward * (forward * v).sum(-1, keepdim=True) * (1 - roll.cos())[None, :, None])
    rotation = torch.stack((rotate(f1), rotate(l1), rotate(up.expand_as(forward))), dim=-1)
    translation = translation[:, None].expand(-1, len(tilt), -1).reshape(options.candidates, 3)
    rotation = rotation.reshape(options.candidates, 3, 3)
    # Upstream uses sampled reach and a soft camera-center term ahead of IK.
    # Keep all candidate lanes and use these terms as preferences, including when nothing fits.
    d = torch.linalg.vector_norm(translation[:, None, None] - p[None], dim=-1)
    furthest = torch.where(valid, d, 0).amax(dim=(1, 2))
    nearest = torch.where(valid, d, torch.inf).amin(dim=(1, 2))
    mean = torch.where(valid, d, 0).sum(dim=(1, 2)) / count.clamp_min(1)
    rel = translation
    image_radius = rel[:, :2].abs().amax(dim=-1) / rel[:, 2].clamp_min(.05)
    occlusion = torch.where(rel[:, 2] > .05, (1 - image_radius).clamp_min(0), 0)
    # Soft placement prior only: does not prove that an in-view base touches a surface.
    lower_edge = (0.55 - rel[:, 1] / rel[:, 2].clamp_min(.05)).clamp(0, 1)
    placement = options.placement * occlusion * (1 + lower_edge)
    bias = (placement + .02 * (mean / group.reach - .65).abs()
            + .08 * (furthest / group.reach - .90).clamp_min(0)
            + (.08 - nearest).clamp_min(0) + (.2 - rel.norm(dim=-1)).clamp_min(0))
    if box is not None and state["frustum"] is not None:
        horizontal, vertical = state["frustum"]
        corners = box @ rotation.transpose(-1, -2) + translation[:, None]
        bx, by, bz = corners.unbind(-1)
        # Match Render's centered camera / square pixels. Penalize the base body
        # inside the image; a base clipping the image edge is not a rejection.
        depth = bz.clamp_min(.05)
        penetration = torch.minimum(1 - (bx / (horizontal * depth)).abs(),
                                    1 - (by / (vertical * depth)).abs())
        penetration = torch.where(bz > .05, penetration.clamp_min(0), 0)
        bias += options.frustum * penetration.mean(dim=-1)
    return rotation, translation, torch.where(count > 0, bias, 0)


class _Arm:
    def __init__(self, group, options):
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState

        self.fk = Kinematics(KinematicsCfg.from_robot_yaml_file(group.config, tool_frames=list(group.tools)),
                             compute_jacobian=True, compute_spheres=False)
        self.state_type = JointState
        self.names = list(self.fk.joint_names)
        self.tools = [list(self.fk.tool_frames).index(name) for name in group.tools]
        order = [list(group.joints).index(name) for name in self.names]
        self.export = [self.names.index(name) for name in group.joints]
        self.home = _gpu(group.home[order])
        self.low, self.high = _gpu(group.lower[order]), _gpu(group.upper[order])
        self.eye = torch.eye(len(order), device="cuda") * DAMPING**2
        self.motion_penalty = options.motion_penalty
        limits = self.fk.get_joint_limits()
        order = [list(limits.joint_names).index(n) for n in self.names]
        self.velocity = torch.minimum(-limits.velocity[0, order], limits.velocity[1, order]).clamp_max(JOINT_SPEED_CEILING)
        self.acceleration = torch.minimum(-limits.acceleration[0, order], limits.acceleration[1, order])
        asset_order = [group.joints.index(n) for n in self.names]
        self.rated_velocity = (self.velocity if group.rated_velocity is None else
                               torch.minimum(self.velocity, _gpu(group.rated_velocity[asset_order])))
        self.rated_acceleration = (self.acceleration if group.rated_acceleration is None else
                                   torch.minimum(self.acceleration, _gpu(group.rated_acceleration[asset_order])))
        self.unit = self.eye / DAMPING**2
        self.weight = _gpu((1, 1, 1, options.orientation, options.orientation, options.orientation))

    def kinematics(self, q, jacobian=True):
        self.fk.compute_jacobian = jacobian
        state = self.fk.compute_kinematics(self.state_type.from_position(q.contiguous(), joint_names=self.names))
        return (state.tool_poses.position[:, 0, self.tools], state.tool_poses.quaternion[:, 0, self.tools],
                state.tool_jacobians[:, 0, self.tools] if jacobian else None)

    def residual(self, q, p, r):
        actual_p, actual_r, jac = self.kinematics(q)
        return p - actual_p, _rotation_error(r, actual_r), jac

    def solve(self, q, p, r, valid, iterations, bound=None, penalty=0.):
        weight = self.weight * valid[..., None]
        start = q
        lower = self.low if bound is None else torch.maximum(start - bound, self.low)
        upper = self.high if bound is None else torch.minimum(start + bound, self.high)
        best, best_error = q, None
        for _ in range(iterations):
            ep, er, jac = self.residual(q, p, r)
            error = (torch.cat((ep, er), -1) * weight).flatten(1)
            cost = error.square().sum(-1, keepdim=True) + penalty * (q - start).square().sum(-1, keepdim=True)
            if best_error is not None:
                best = torch.where(cost < best_error, q, best)
                best_error = torch.minimum(best_error, cost)
            else:
                best_error = cost
            j = (jac * weight[..., None]).flatten(1, 2)
            jt = j.transpose(1, 2)
            normal = jt @ j + self.eye * (1 + penalty / DAMPING**2)
            rhs = jt @ error[..., None] + penalty * (start - q)[..., None]
            # solve_ex avoids a host synchronization for info checking.
            step = torch.linalg.solve_ex(normal, rhs, check_errors=False).result.squeeze(-1)
            q = (q + step.nan_to_num().clamp(-SEARCH_JOINT_STEP, SEARCH_JOINT_STEP)).clamp(lower, upper)
        ep, er, _ = self.residual(q, p, r)
        cost = ((torch.cat((ep, er), -1) * weight).flatten(1).square().sum(-1, keepdim=True)
                + penalty * (q - start).square().sum(-1, keepdim=True))
        q = torch.where(cost < best_error, q, best)
        ep, er, _ = self.residual(q, p, r)
        return q, ep.norm(dim=-1), er.norm(dim=-1)


    def track(self, start, p, r, valid, iterations, seconds, previous, mode):
        """Box-constrained LM with feasible trial steps; each row is one arm and direction."""
        movement = (self.rated_velocity if mode == "rated" else self.velocity) * seconds
        has_target = valid.any(-1, keepdim=True)
        if mode in ("braked", "rated"):
            acceleration = self.rated_acceleration if mode == "rated" else self.acceleration
            a = (acceleration * seconds.square()).clamp_min(1e-6)
            # A sufficient discrete stopping envelope: d + d²/(2a) <= distance.
            # It reserves space to stop on subsequent frames without hitting the joint limit.
            back = (a.square() + 2 * a * (start - self.low).clamp_min(0)).sqrt() - a
            fwd = (a.square() + 2 * a * (self.high - start).clamp_min(0)).sqrt() - a
            delta_min = torch.maximum(-torch.minimum(movement, back), previous - a)
            delta_max = torch.minimum(torch.minimum(movement, fwd), previous + a)
            # Both branches start at rest inside the authored range. Protect against
            # ~1e-7 floating-point discrepancy at a single-point interval.
            delta_min = torch.minimum(delta_min, delta_max)
            low = torch.maximum(self.low, start + delta_min)
            high = torch.minimum(self.high, start + delta_max)
        else:
            movement = torch.where(has_target, movement, 0)
            low = torch.maximum(self.low, start - movement)
            high = torch.minimum(self.high, start + movement)
        # Penalize acceleration without forcing a joint to move through a limit to decelerate.
        accel_weight = ACCELERATION_WEIGHT if mode in ("smooth", "pose", "braked", "rated") else 0.
        predict = start + torch.where(has_target, previous, 0)
        scale = self.weight if mode != "pose" else self.weight * _gpu((1., 1., 1., 1.6, 1.6, 1.6))
        weight = scale * valid[..., None]
        penalty = self.motion_penalty
        reg = (DAMPING**2 + penalty + accel_weight) * self.unit

        def cost(ep, er, q):
            residual = torch.cat((ep, er), -1) * weight
            return (residual.square().sum((-2, -1)) + penalty * (q - start).square().sum(-1)
                    + accel_weight * (q - predict).square().sum(-1))

        def descent(q, error, jac):
            jt = jac.transpose(1, 2)
            grad = (jt @ error[..., None]).squeeze(-1) + penalty * (start-q) + accel_weight * (predict-q)
            blocked = ((q <= low + 1e-6) & (grad <= 0)) | ((q >= high - 1e-6) & (grad >= 0))
            free = ~blocked
            j = jac * free[:, None, :]
            h = j.transpose(1, 2) @ j + reg
            g = (grad * free)[..., None]
            gn = torch.linalg.solve_ex(h, g, check_errors=False).result.squeeze(-1).nan_to_num()
            # A feasible gradient step can improve the true objective when a clipped
            # coupled Gauss–Newton step does not.
            denom = (g.transpose(1, 2) @ h @ g).reshape(-1, 1).clamp_min(1e-12)
            pg = g.squeeze(-1) * g.squeeze(-1).square().sum(-1, keepdim=True) / denom
            return gn.clamp(-SEARCH_JOINT_STEP, SEARCH_JOINT_STEP), pg.clamp(-SEARCH_JOINT_STEP, SEARCH_JOINT_STEP)

        q = start.clamp(low, high)
        for _ in range(iterations):
            ep, er, jac = self.residual(q, p, r)
            err = (torch.cat((ep, er), -1) * weight).flatten(1)
            j = (jac * weight[..., None]).flatten(1, 2)
            gn, pg = descent(q, err, j)
            trials = torch.stack((q + gn, q + .5 * gn, q + pg)).clamp(low, high)
            x, w, _ = self.kinematics(trials.flatten(0, 1), jacobian=False)
            ep1 = p - x.unflatten(0, (3, len(q)))
            er1 = _rotation_error(r.expand(3, -1, -1, -1), w.unflatten(0, (3, len(q))))
            all_costs = torch.cat((cost(ep, er, q)[None], cost(ep1, er1, trials)))
            all_q = torch.cat((q[None], trials))
            choice = all_costs.argmin(0)[None, :, None].expand(1, -1, q.shape[-1])
            q = all_q.gather(0, choice).squeeze(0)
        x, w, _ = self.kinematics(q, jacobian=False)
        return q, (p - x).norm(dim=-1), _rotation_error(r, w).norm(dim=-1)


class KeyframeBackend:
    def __init__(self, robot, capacity, options=None):
        self.robot = robot
        self.arms = {}
        self.keys = [g.batch_key or ("group", g.name) for g in robot.groups]
        self.capacity = capacity
        self.options = options or SearchOptions()
        self.boxes = {}
        if self.options.frustum:
            from egowhale.action.base_ik import base_boxes

            self.boxes = {name: _gpu(points) for name, points in base_boxes(robot).items()}

    def reset(self):
        self.arms.clear()

    def _arm(self, index):
        key = self.keys[index]
        if key not in self.arms:
            self.arms[key] = _Arm(self.robot.groups[index], self.options)
        return self.arms[key]

    @torch.inference_mode()
    def prepare(self, data):
        from curobo._src.geom.transform import matrix_to_quaternion

        data_fps(data)
        length = len(data["position"])
        if length < 1:
            raise ValueError("empty episode")
        p, r = _gpu(data["position"]), _gpu(data["rotation"])
        valid = torch.as_tensor(data["valid"], device=p.device, dtype=torch.bool)
        # Invalid hand values can be NaN: replace them before using matrix algebra.
        valid &= p.isfinite().all(-1) & r.isfinite().all((-2, -1))
        p = torch.where(valid[..., None], p, 0)
        r = torch.where(valid[..., None, None], r, torch.eye(3, device=p.device))
        score = (p * r[..., :, 0]).sum(-1)
        swap = ((score < 0) & valid).sum() > .8 * valid.sum()
        frustum = None
        if self.boxes and data.get("camera_size") is not None and data.get("intrinsic") is not None:
            h, w = data["camera_size"]
            f = float(data["intrinsic"][1, 1])
            if f > 0:
                frustum = (.5 * w / f, .5 * h / f)
        return dict(data=data, frame=0, length=length, p=p, r=r, valid=valid, swap=swap, frustum=frustum,
                    keys=_keyframes(p, r, valid, self.options.keys), groups=None, to_quat=matrix_to_quaternion)

    def _goals(self, state, index, frames, rotation, translation):
        group = self.robot.groups[index]
        p, r = state["p"][frames][:, group.sides], state["r"][frames][:, group.sides]
        valid = state["valid"][frames][:, group.sides]
        inverse = rotation.transpose(-1, -2)
        p = torch.einsum("bij,kbtj->kbti", inverse, p[:, None] - translation[None, :, None])
        r = inverse[None, :, None] @ r[:, None]
        q = state["to_quat"](r.contiguous().reshape(-1, 3, 3)).reshape(*r.shape[:-2], 4)
        return p, q, valid[:, None].expand(-1, len(rotation), -1)

    def _solve_rows(self, rows, current, iterations, bound=None, motion=None):
        """Share a single FK batch for structurally identical SolveGroups."""
        buckets, result = {}, {}
        for key, (p, r, v) in rows.items():
            buckets.setdefault(self.keys[key[1]], []).append((key, p, r, v))
        for entries in buckets.values():
            arm = self._arm(entries[0][0][1])
            p, r, v = (torch.cat([row[j] for row in entries]) for j in (1, 2, 3))
            q = torch.cat([current[key] for key, *_ in entries])
            lengths = [len(row[1]) for row in entries]
            if motion is None or self.options.tracking == "damped":
                columns = arm.solve(q, p, r, v, iterations, bound, arm.motion_penalty if bound is not None else 0.)
            else:
                dt = _gpu([motion[key][0] for key, *_ in entries]).repeat_interleave(lengths[0])[:, None]
                previous = torch.cat([motion[key][1] for key, *_ in entries])
                columns = arm.track(q, p, r, v, iterations, dt, previous, self.options.tracking)
            result.update(zip((row[0] for row in entries), zip(*(x.split(lengths) for x in columns))))
        return result

    def _search(self, states, hypotheses):
        goals, current, worst_p, worst_r = {}, {}, {}, {}
        blended = self.options.tail_mix < 1.
        sum_p, sum_r, count = {}, {}, {}
        for (s, g), (rotation, translation) in hypotheses.items():
            goals[s, g] = self._goals(states[s], g, states[s]["keys"], rotation, translation)
            current[s, g] = self._arm(g).home.expand(len(rotation), -1)
            worst_p[s, g] = rotation.new_zeros(len(rotation))
            worst_r[s, g] = rotation.new_zeros(len(rotation))
            if blended:
                sum_p[s, g] = rotation.new_zeros(len(rotation))
                sum_r[s, g] = rotation.new_zeros(len(rotation))
                count[s, g] = rotation.new_zeros(len(rotation))
        for k in range(self.options.keys):
            rows = {key: tuple(x[k] for x in goal) for key, goal in goals.items()}
            for key, (q, ep, er) in self._solve_rows(rows, current, self.options.search_steps).items():
                current[key] = q
                v = rows[key][-1]
                worst_p[key] = torch.maximum(worst_p[key], torch.where(v, ep, 0).amax(-1))
                worst_r[key] = torch.maximum(worst_r[key], torch.where(v, er, 0).amax(-1))
                if blended:
                    frames = states[key[0]]["keys"]
                    if k:
                        v = v & (frames[k] != frames[k-1])
                    sum_p[key] += torch.where(v, ep, 0).sum(-1)
                    sum_r[key] += torch.where(v, er, 0).sum(-1)
                    count[key] += v.sum(-1)
        if not blended:
            return {key: (p, worst_r[key]) for key, p in worst_p.items()}
        a = self.options.tail_mix
        return {key: (a * p + (1-a) * sum_p[key] / count[key].clamp_min(1),
                      a * worst_r[key] + (1-a) * sum_r[key] / count[key].clamp_min(1))
                for key, p in worst_p.items()}

    def _choose(self, states):
        groups = list(self.robot.groups)
        shared = len(groups) == 1 and len(groups[0].sides) > 1
        candidates = {(s, g): _candidates(state, group, -1. if group.sides == (0,) else 1., shared, self.options, self.boxes.get(group.name))
                      for s, state in enumerate(states) for g, group in enumerate(groups)}
        solo = self._search(states, {key: value[:2] for key, value in candidates.items()})
        top = {key: (solo[key][0] + .05 * solo[key][1] + bias).topk(self.options.top, largest=False).indices
               for key, (_, _, bias) in candidates.items()}
        combinations = list(product(range(self.options.top), repeat=len(groups))) if len(groups) > 1 else None
        pair_scores, pair_ok, has_pair, placement = {}, {}, {}, {}
        if combinations is not None:
            indices = torch.tensor(combinations, device="cuda", dtype=torch.long).T
            for s, state in enumerate(states):
                for g in range(len(groups)):
                    rows = top[s, g][indices[g]]
                    pair_scores[s, g] = tuple(x[rows] for x in candidates[s, g][:2])
                    placement[s, g] = candidates[s, g][2][rows]
                pair_ok[s] = torch.ones(len(combinations), device="cuda", dtype=torch.bool)
                for left, right in ((i, j) for i in range(len(groups)) for j in range(i+1, len(groups))):
                    if groups[left].sides[0] > groups[right].sides[0]:
                        left, right = right, left
                    separation = pair_scores[s, right][1] - pair_scores[s, left][1]
                    minimum = max(.25, min(.25 * (groups[left].reach + groups[right].reach), .5))
                    side = separation[:, 0] * torch.where(state["swap"], -1., 1.)
                    relevant = state["valid"][:, groups[left].sides].any() & state["valid"][:, groups[right].sides].any()
                    pair_ok[s] &= (~relevant) | ((separation.norm(dim=-1) >= minimum) & (side >= .05))
            paired = self._search(states, pair_scores)
            for s in range(len(states)):
                # Match upstream: worst positional and angular errors across both arms.
                positions = torch.stack([paired[s, g][0] for g in range(len(groups))])
                rotations = torch.stack([paired[s, g][1] for g in range(len(groups))])
                prior = torch.stack([placement[s, g] for g in range(len(groups))]).mean(0)
                combined = positions.amax(0) + .05 * rotations.amax(0) + prior
                has_pair[s] = pair_ok[s].any()
                pair_ok[s] = torch.where(has_pair[s], combined.masked_fill(~pair_ok[s], torch.inf), combined).argmin().reshape(1)
        for s, state in enumerate(states):
            state["pair_valid"] = has_pair.get(s, torch.ones((), device="cuda", dtype=torch.bool))
            state["groups"] = []
            for g in range(len(groups)):
                if combinations is None:
                    index = top[s, g][:1]
                    rotation, translation = (x[index] for x in candidates[s, g][:2])
                else:
                    rotation, translation = (x[pair_ok[s]] for x in pair_scores[s, g])
                frames = torch.arange(state["length"], device="cuda")
                p, r, valid = (x[:, 0] for x in self._goals(state, g, frames, rotation, translation))
                arm = self._arm(g)
                base = torch.eye(4, device="cuda")
                base[:3, :3], base[:3, 3] = rotation[0], translation[0]
                directions = 2 if self.options.bidirectional else 1
                state["groups"].append(dict(base=base, p=p, r=r, valid=valid, q=arm.home.expand(directions, -1),
                                            velocity=arm.home.new_zeros(directions, len(arm.names)),
                                            q_out=p.new_empty((directions, len(p), len(arm.names))),
                                            errs=p.new_empty((directions, len(p), p.shape[1], 2))))
        self._warmup(states)

    def _warmup(self, states):
        tasks, current, homes, targets = {}, {}, {}, {}
        for s, state in enumerate(states):
            for g, task in enumerate(state["groups"]):
                key = s, g
                tasks[key] = task
                current[key] = task["q"]
                hp, hq, _ = self._arm(g).kinematics(task["q"])
                homes[key] = hp, hq
                first = task["valid"].long().argmax(dim=0, keepdim=True)
                if self.options.bidirectional:
                    last = len(task["p"]) - 1 - task["valid"].long().flip(0).argmax(dim=0, keepdim=True)
                    first = torch.cat((first, last))
                tools = torch.arange(task["p"].shape[1], device="cuda")
                p, r = task["p"][first, tools], task["r"][first, tools]
                r = torch.where((r * hq).sum(-1, keepdim=True) < 0, -r, r)
                targets[key] = p, r, task["valid"].any(dim=0, keepdim=True).expand_as(first)
        for i in range(1, WARMUP_STEPS + 1):
            a = i / WARMUP_STEPS
            rows = {key: (hp.lerp(targets[key][0], a), F.normalize(hq.lerp(targets[key][1], a), dim=-1),
                          targets[key][2]) for key, (hp, hq) in homes.items()}
            for key, (q, _, _) in self._solve_rows(rows, current, 2).items():
                current[key] = q
        for key, task in tasks.items():
            task["q"] = current[key]

    @torch.inference_mode()
    def step(self, states):
        uninitialized = [s for s in states if s["groups"] is None]
        if uninitialized:
            self._choose(uninitialized)
        rows, current, motion = {}, {}, {}
        for s, state in enumerate(states):
            f = state["frame"]
            dt = 1. / data_fps(state["data"])
            reverse = state["length"] - 1 - f
            for g, task in enumerate(state["groups"]):
                rows[s, g] = tuple(torch.cat((task[k][f:f+1], task[k][reverse:reverse+1]))
                                   if self.options.bidirectional else task[k][f:f+1]
                                   for k in ("p", "r", "valid"))
                current[s, g] = task["q"]
                if self.options.tracking != "damped":
                    motion[s, g] = (dt, task["velocity"])
        for (s, g), (q, ep, er) in self._solve_rows(rows, current, self.options.track_steps, self.options.track_step, motion).items():
            state = states[s]
            task, f = state["groups"][g], state["frame"]
            if self.options.tracking != "damped":
                task["velocity"] = q - task["q"]
            task["q"] = q
            task["q_out"][0, f] = q[0]
            task["errs"][0, f, :, 0], task["errs"][0, f, :, 1] = ep[0], er[0]
            if self.options.bidirectional:
                rev = state["length"] - 1 - f
                task["q_out"][1, rev] = q[1]
                task["errs"][1, rev, :, 0], task["errs"][1, rev, :, 1] = ep[1], er[1]
        for state in states:
            state["frame"] += 1

    @torch.inference_mode()
    def finish(self, state):
        groups, data = list(self.robot.groups), state["data"]
        fields = [state["pair_valid"].float().reshape(1)]
        for g, task in enumerate(state["groups"]):
            valid = task["valid"]
            count = valid.sum().clamp_min(1)
            candidates = task["q_out"]
            residuals = task["errs"]
            if self.options.bidirectional:
                # Select one complete trajectory per independent group. No frame-by-frame
                # branch switching or CPU synchronization, including in the decision.
                pe, re = residuals.unbind(-1)
                ep = (pe * valid).sum((1, 2)) / count
                er = (re * valid).sum((1, 2)) / count
                bad = ((pe > .05) & valid).sum((1, 2)) / count
                step = candidates.diff(dim=1).abs().amax(dim=-1)
                jump = (step > 5.5 / data_fps(data)).sum(1) / max(step.shape[1], 1)
                choice = (ep + .022 * er + .18 * bad + .12 * jump).argmin().reshape(1)
                q = candidates.index_select(0, choice)[0]
                errors = residuals.index_select(0, choice)[0]
            else:
                q, errors = candidates[0], residuals[0]
            q = q[:, self._arm(g).export]
            mean = (errors * valid[..., None]).sum((0, 1)) / count
            worst = torch.where(valid[..., None], errors, 0).amax((0, 1))
            jerk = q.diff(n=3, dim=0).abs().mean() if len(q) >= 4 else q.new_zeros(())
            failure = ((errors[..., 0] > .05) & valid).sum()
            wrong_rotation = ((errors[..., 1] > torch.pi / 6) & valid).sum()
            counts = torch.stack((valid.sum(), failure, wrong_rotation)).float()
            fields.extend((task["base"].flatten(), q.flatten(), mean, worst, jerk.reshape(1), counts))
        # Only selected bases, final motions and their metrics cross the device boundary.
        packed = torch.cat(fields).cpu().numpy()
        if not np.isfinite(packed).all():
            raise ValueError("non-finite final IK result")
        cursor, bases, trajectories, losses = 1, {}, [], {}
        for group in groups:
            bases[group.name] = packed[cursor:cursor+16].reshape(4, 4)
            cursor += 16
            stop = cursor + state["length"] * len(group.joints)
            trajectories.append(packed[cursor:stop].reshape(-1, len(group.joints)))
            losses[group.name] = dict(zip(("position", "orientation", "max_position", "max_orientation", "jerk"),
                                         map(float, packed[stop:stop+5]), strict=True))
            losses[group.name].update(zip(("valid_steps", "position_fail_steps", "orientation_fail_steps"),
                                          map(int, packed[stop+5:stop+8]), strict=True))
            cursor = stop + 8
        return {**self.robot.pack(trajectories, data["width"], data["valid"]), "fps": data_fps(data), "bases": bases,
                "base_frame": "camera", "pair_constraints_met": bool(packed[0]),
                "motion_model": self.options.tracking,
                "group_losses": losses,
                "losses": {key: float(np.mean([v[key] for v in losses.values()])) for key in next(iter(losses.values()))}}
