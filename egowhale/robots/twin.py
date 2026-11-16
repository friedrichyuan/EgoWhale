"""Independent copies of a single-arm MJCF, with matching cuRobo kinematics."""

import json
import os
import warnings
import xml.etree.ElementTree as ET
from functools import cached_property

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egowhale.paths import ASSETS

from .base import Robot, SolveGroup

_SIDES = ("left", "right")
# Retarget uses x for approach and z for finger separation. Both source grippers
# extend along local +z and close along local y.
_TCP_ROTATION = Rotation.from_matrix(np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]]))
_SEARCH_BOUNDS = np.array([.30, .30, .20, np.deg2rad(35.)])


def _origin(joint, position, rotation):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
        rpy = rotation.as_euler("xyz")
    ET.SubElement(joint, "origin", xyz=" ".join(map(str, position)), rpy=" ".join(map(str, rpy)))


class TwinMJCF(Robot):
    tcp_sites = ("left_tcp", "right_tcp")

    @property
    def roots(self):
        return {side: f"{side}_{self.base_body}" for side in _SIDES}

    def _source_spec(self):
        source = mujoco.MjSpec.from_file(str(ASSETS / "robots" / self.name / "mjcf" / self.xml))
        site = source.site("attachment_site")
        site.name = "tcp"
        site.quat = _TCP_ROTATION.as_quat()[[3, 0, 1, 2]]
        return source

    @cached_property
    def source_model(self):
        return self._source_spec().compile()

    def scene_spec(self):
        source = self._source_spec()
        # Per-arm home poses are read from the source model. The assembled
        # scene is posed from exported joints and needs no keyframes.
        for key in list(source.keys):
            source.delete(key)
        scene = mujoco.MjSpec()
        scene.option.integrator = source.option.integrator
        scene.option.impratio = source.option.impratio
        scene.copy_during_attach = True
        frame = scene.worldbody.add_frame()
        for side in _SIDES:
            scene.attach(source, prefix=f"{side}_", frame=frame)
        scene.worldbody.add_camera(name=self.camera)
        scene.worldbody.add_light(pos=[0., -1., 1.], dir=[0., 1., -1.])
        return scene

    def _urdf(self, side):
        model = self.source_model
        site = model.site("tcp")
        base = model.body(self.base_body).id
        chain = []
        body = model.body(int(site.bodyid[0]))
        while body.id != base:
            if body.id == 0:
                raise ValueError(f"{self.name}: TCP is outside the arm chain")
            chain.append(body)
            body = model.body(int(body.parentid[0]))

        prefix = f"{side}_"
        root = ET.Element("robot", name=prefix + self.name)
        ET.SubElement(root, "link", name=prefix + self.base_body)
        for body in reversed(chain):
            name = prefix + body.name
            ET.SubElement(root, "link", name=name)
            joint = model.joint(int(body.jntadr[0])) if body.jntnum[0] else None
            if joint is not None and (
                body.jntnum[0] != 1 or joint.name not in self.arm_joints
                or joint.type[0] != mujoco.mjtJoint.mjJNT_HINGE or np.any(joint.pos)
                or model.qpos0[joint.qposadr[0]] != 0
            ):
                raise ValueError(f"{self.name}: unsupported arm joint on {body.name}")
            node = ET.SubElement(root, "joint", name=prefix + (joint.name if joint else body.name + "_fixed"),
                                 type="revolute" if joint else "fixed")
            ET.SubElement(node, "parent", link=prefix + model.body(int(body.parentid[0])).name)
            ET.SubElement(node, "child", link=name)
            _origin(node, body.pos, Rotation.from_quat(body.quat[[1, 2, 3, 0]]))
            if joint is not None:
                ET.SubElement(node, "axis", xyz=" ".join(map(str, joint.axis)))
                ET.SubElement(node, "limit", lower=str(joint.range[0]), upper=str(joint.range[1]),
                              effort="100", velocity="5.5")
        ET.SubElement(root, "link", name=prefix + "tcp")
        tool = ET.SubElement(root, "joint", name=prefix + "tool_fixed", type="fixed")
        ET.SubElement(tool, "parent", link=prefix + model.body(int(site.bodyid[0])).name)
        ET.SubElement(tool, "child", link=prefix + "tcp")
        _origin(tool, site.pos, Rotation.from_quat(site.quat[[1, 2, 3, 0]]))
        return ET.tostring(root, encoding="utf-8")

    @cached_property
    def groups(self):
        model = self.source_model
        home = model.key("home").qpos[[model.joint(n).qposadr[0] for n in self.arm_joints]]
        result = []
        self._urdf_files = []
        for i, side in enumerate(_SIDES):
            joints = tuple(f"{side}_{joint}" for joint in self.arm_joints)
            tools = (self.tcp_sites[i],)
            lower, upper = self.limits(joints)
            # Keep the anonymous URDF alive for cuRobo while this adapter lives.
            urdf = self._urdf(side)
            fd = os.memfd_create(f"{side}_{self.name}.urdf", os.MFD_CLOEXEC)
            handle = os.fdopen(fd, "w+b")
            handle.write(urdf)
            handle.flush()
            self._urdf_files.append(handle)
            config = {"robot_cfg": {"kinematics": {
                "urdf_path": f"/proc/self/fd/{fd}", "asset_root_path": "",
                "base_link": self.roots[side], "tool_frames": list(tools),
                "collision_link_names": None, "collision_spheres": None,
                "cspace": {"joint_names": list(joints), "default_joint_position": home.tolist(),
                           "null_space_weight": [1.] * len(joints), "cspace_distance_weight": [1.] * len(joints),
                           "max_acceleration": 15., "max_jerk": 500.}}}}
            result.append(SolveGroup(side, (i,), joints, tools, config, lower, upper, home.copy(),
                                     self.reach, _SEARCH_BOUNDS.copy(), batch_key=self.name))
        return tuple(result)

    def body_side(self, name):
        return next((side for side in _SIDES if name.startswith(f"{side}_")), None)

    def is_gripper(self, model, geom_id):
        if model.geom_group[geom_id] != 2:
            return False
        name = model.body(int(model.geom_bodyid[geom_id])).name
        side = self.body_side(name)
        return side is not None and name.removeprefix(f"{side}_") in self.gripper_bodies

    def hide(self, model):
        model.geom_rgba[model.geom_group == 3, 3] = 0.
        model.site_rgba[:, 3] = 0.

    def quality_model(self):
        return self.scene_spec().compile()


class FrankaRobot(TwinMJCF):
    name = "franka"
    xml = "panda.xml"
    base_body = "link0"
    arm_joints = tuple(f"joint{i}" for i in range(1, 8))
    gripper_bodies = ("left_finger", "right_finger")
    reach = .955

    def gripper_positions(self, width, valid):
        opening = np.where(valid, np.clip(width / 2., 0., .04), 0.)
        return {f"{side}_finger_joint{j}": opening[:, i] for i, side in enumerate(_SIDES) for j in (1, 2)}


class UR5eRobot(TwinMJCF):
    name = "ur5e"
    xml = "ur5e.xml"
    base_body = "ur5e/base"
    arm_joints = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                  "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
    gripper_bodies = tuple(f"{side}_{body}" for side in _SIDES for body in
                           ("driver", "coupler", "spring_link", "follower", "pad", "silicone_pad"))
    reach = .99

    @cached_property
    def _gripper_calibration(self):
        path = ASSETS / "robots" / self.name / "robotiq_2f85_calibration.json"
        cal = json.loads(path.read_text())
        names = cal["joint_names"]
        model = self.source_model
        ranges = np.array([model.joint(n).range for n in names])
        qpos = np.clip(cal["qpos"], ranges[:, :1], ranges[:, 1:])
        # Invert the sampled linkage using the pad opening in metres.
        data = mujoco.MjData(model)
        data.qpos[:] = model.key("home").qpos
        addresses = [model.joint(n).qposadr[0] for n in names]
        gaps = []
        for sample in qpos.T:
            data.qpos[addresses] = sample
            mujoco.mj_kinematics(model, data)
            axis = data.body("base").xmat.reshape(3, 3)[:, 1]
            gaps.append(min(
                (data.geom(f"right_pad{i}").xpos - data.geom(f"left_pad{i}").xpos) @ axis
                - model.geom(f"right_pad{i}").size[1] - model.geom(f"left_pad{i}").size[1]
                for i in (1, 2)
            ))
        order = np.argsort(gaps)
        return names, np.asarray(gaps)[order], qpos[:, order]

    def gripper_positions(self, width, valid):
        names, gaps, qpos = self._gripper_calibration
        opening = np.where(valid, width, 0.)
        return {f"{side}_{joint}": np.interp(opening[:, i], gaps, samples)
                for i, side in enumerate(_SIDES) for joint, samples in zip(names, qpos, strict=True)}
