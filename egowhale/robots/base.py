"""Robot asset adapters: no episode I/O, scheduling, or numerical solvers."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

CAMERA_ROTATION = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])


@dataclass
class SolveGroup:
    name: str
    sides: tuple
    joints: tuple
    tools: tuple
    config: object
    lower: np.ndarray
    upper: np.ndarray
    home: np.ndarray
    reach: float
    bounds: np.ndarray

    def seed(self, position, valid):
        usable = valid.all(axis=1) if valid.all(axis=1).any() else valid.any(axis=1)
        if not usable.any():
            raise ValueError(f"{self.name}: no valid target frames")
        midpoint = position[usable].mean(axis=1).mean(axis=0)
        translation = midpoint - CAMERA_ROTATION @ np.array([0.65 * self.reach, 0., 0.15])
        return CAMERA_ROTATION, translation


class Robot(ABC):
    camera = "ego_camera_calibrated"

    @abstractmethod
    def scene_spec(self):
        """Return a fresh MuJoCo MjSpec from the downloaded assets."""

    @cached_property
    def model(self):
        return self.scene_spec().compile()

    @cached_property
    def joint_names(self):
        return tuple(self.model.joint(i).name for i in range(self.model.njnt))

    @cached_property
    @abstractmethod
    def groups(self):
        """Independent base/IK and approach groups."""

    @abstractmethod
    def gripper_positions(self, width, valid):
        """Return {joint_name: per-frame positions}."""

    @abstractmethod
    def body_side(self, name):
        """Return left/right/shared for robot bodies, None for other bodies."""

    @abstractmethod
    def is_gripper(self, model, geom_id):
        pass

    @abstractmethod
    def hide(self, model):
        pass

    @property
    @abstractmethod
    def roots(self):
        """Map solve-group names to scene root bodies."""

    @property
    @abstractmethod
    def tcp_sites(self):
        """Sites in retarget left/right order."""

    def limits(self, names):
        return np.array([self.model.joint(n).range for n in names]).T

    def pack(self, arms, width, valid):
        values = {name: arm[:, i] for group, arm in zip(self.groups, arms) for i, name in enumerate(group.joints)}
        values.update(self.gripper_positions(width, valid))
        return {"joint_names": np.asarray(self.joint_names),
                "joint_positions": np.stack([values[n] for n in self.joint_names], axis=1).astype(np.float32)}

    def qpos(self, motion, model=None):
        model = self.model if model is None else model
        # Read pre-adapter ARX outputs as well as new named trajectories.
        if "joint_positions" not in motion:
            return np.asarray(motion["qpos"])
        values = motion["joint_positions"]
        out = np.zeros((len(values), model.nq), dtype=values.dtype)
        for col, name in enumerate(motion["joint_names"]):
            out[:, model.joint(str(name)).qposadr[0]] = values[:, col]
        return out

    def named(self, qpos):
        columns = [self.model.joint(n).qposadr[0] for n in self.joint_names]
        return {"joint_names": np.asarray(self.joint_names), "joint_positions": qpos[:, columns]}

    def place_bases(self, model, bases):
        if isinstance(bases, np.ndarray):
            bases = {self.groups[0].name: bases}
        for group, matrix in bases.items():
            matrix = np.asarray(matrix)
            body = model.body(self.roots[group])
            body.pos[:] = matrix[:3, 3]
            body.quat[:] = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def place_camera(self, model, intrinsic, height):
        cam = model.camera(self.camera)
        cam.pos[:] = 0.
        cam.quat[:] = Rotation.from_matrix(np.diag([1., -1., -1.])).as_quat()[[3, 0, 1, 2]]
        cam.fovy[0] = float(np.degrees(2. * np.arctan(height / (2. * float(intrinsic[1, 1])))))

    def quality_model(self):
        spec = self.scene_spec()
        for body in spec.bodies:
            if self.body_side(body.name) is None:
                continue
            for geom in list(body.geoms):
                if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                    body.add_geom(name=f"{geom.name}_hit", type=mujoco.mjtGeom.mjGEOM_MESH,
                                  meshname=geom.meshname, pos=geom.pos, quat=geom.quat,
                                  contype=1, conaffinity=1, group=3, rgba=[0., 0., 0., 0.])
        return spec.compile()
