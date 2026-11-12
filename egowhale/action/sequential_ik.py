"""cuRobo numerical backend for episode streams. No I/O or scheduling dependencies."""

import numpy as np
import torch

from egowhale.action import base_ik as core


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
        self.solvers, self.kinematics = {}, {}

    def reset(self):
        self.solvers.clear()
        self.kinematics.clear()
        import gc

        gc.collect()

    def prepare(self, data):
        core.data_fps(data)  # Validate before allocating any GPU state.
        length = len(data["position"])
        if length < 1:
            raise ValueError("empty episode")
        tasks = []
        for group, key in zip(self.groups, self.keys, strict=True):
            sides = list(group.sides)
            valid = data["valid"][:, sides]
            position, rotation = data["position"][:, sides], data["rotation"][:, sides]
            seed_r, seed_t = group.seed(position, valid)
            gp, gr = core._hold(position, rotation, valid)
            offsets = core._offset_grid(group.bounds)
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
                    p=torch.as_tensor(p, device="cuda", dtype=torch.float32),
                    q=torch.as_tensor(q, device="cuda", dtype=torch.float32),
                    current=None,
                    arms=[],
                    errors=[],
                )
            )
        return dict(data=data, length=length, frame=0, tasks=tasks)

    def solver(self, key):
        if key not in self.solvers:
            from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg

            groups = self.buckets[key]
            self.solvers[key] = InverseKinematics(
                InverseKinematicsCfg.create(
                    robot=groups[0].config,
                    num_seeds=core._SEEDS,
                    self_collision_check=False,
                    load_collision_spheres=False,
                    use_cuda_graph=False,
                    max_batch_size=self.batch_widths[key],
                    optimizer_configs=["ik/lbfgs_ik.yml"],
                )
            )
        return self.solvers[key]

    def step(self, states):
        from curobo.types import GoalToolPose, JointState, Pose

        for key, groups in self.buckets.items():
            group = groups[0]
            tasks = [(s, t) for s in states for t in s["tasks"] if t["key"] == key]
            ik = self.solver(key)
            names = list(ik.joint_names)
            order = [list(group.joints).index(n) for n in names]
            p = torch.cat([t["p"][s["frame"]] for s, t in tasks])
            q = torch.cat([t["q"][s["frame"]] for s, t in tasks])
            currents = []
            for s, t in tasks:
                if t["current"] is None:
                    # New episode never inherits the removed episode's state, even when sharing its slot.
                    home = torch.as_tensor(t["group"].home, device=p.device, dtype=p.dtype)
                    currents.append(home[order].repeat(len(t["offsets"]), 1))
                else:
                    currents.append(t["current"])
            current = torch.cat(currents)
            count = len(p)
            width = self.batch_widths[key]
            padding = width - count
            if padding < 0:
                raise ValueError(f"IK batch for {key} has {count} rows, capacity is {width}")
            if padding:
                # Keep cuRobo's cached batch shape fixed even as episodes finish at different times.
                # A matched pose and current state from a real row give the dummy rows valid inputs.
                p = torch.cat((p, p[:1].expand(padding, -1, -1)))
                q = torch.cat((q, q[:1].expand(padding, -1, -1)))
                current = torch.cat((current, current[:1].expand(padding, -1)))
            current = JointState.from_position(current, joint_names=names)
            goals = GoalToolPose.from_poses(
                {
                    name: Pose(position=p[:, side], quaternion=q[:, side], name=name, normalize_rotation=True)
                    for side, name in enumerate(group.tools)
                },
                ordered_tool_frames=list(group.tools),
            )
            result = ik.solve_pose(goal_tool_poses=goals, current_state=current)
            values = result.js_solution.position
            while values.ndim > 2:
                values = values[:, 0]
            result_names = list(result.js_solution.joint_names or names)
            # Only real rows may update a candidate's state or contribute to base selection.
            values = values[:count, [result_names.index(n) for n in names]]
            error = result.position_error.reshape(width, -1)[:count].mean(dim=-1)
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
            arm = torch.stack(task["arms"]).detach().float().cpu().numpy()
            error = torch.stack(task["errors"]).detach().float().cpu().numpy()
            if not np.isfinite(arm).all() or not np.isfinite(error).all():
                raise ValueError("non-finite sequential IK result")
            choice = int(error.mean(axis=0).argmin())
            offset, trajectory, terms = core._refine(
                self.kinematics[key],
                task["gp"],
                task["gr"],
                task["valid"],
                task["seed_r"],
                task["seed_t"],
                task["offsets"][choice],
                arm[:, choice],
                canonical,
                fps=core.data_fps(state["data"]),
            )
            arms.append(core._smooth_arm(trajectory))
            bases[task["group"].name] = core._camera_matrix(task["seed_r"], task["seed_t"], offset)
            losses[task["group"].name] = terms
        data = state["data"]
        return {
            **self.robot.pack(arms, data["width"], data["valid"]),
            "fps": core.data_fps(data),
            "bases": bases,
            "group_losses": losses,
            "losses": {k: float(np.mean([v[k] for v in losses.values()])) for k in next(iter(losses.values()))},
        }
