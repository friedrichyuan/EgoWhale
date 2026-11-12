"""Two independent PiPer instances assembled in memory from the Menagerie source.

No generated assets are installed. cuRobo's file-based URDF reader receives a
process-owned anonymous file descriptor containing kinematics from the same MJCF.
"""
import copy
import os
import warnings
import xml.etree.ElementTree as ET
from functools import cached_property

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egowhale.paths import ASSETS

from .base import Robot, SolveGroup

SOURCE = ASSETS / "piper_sources/menagerie/agilex_piper/piper.xml"
# Centers of the distal inner finger pads in the downloaded model's body frames.
PADS = (("link7", (0., -.045, 0.)), ("link8", (0., .045, 0.)))
BOUNDS = np.array([.30, .30, .20, np.deg2rad(35.)])


def _numbers(values):
    return " ".join(str(float(v)) for v in values)


class PiperRobot(Robot):
    name = "piper"
    roots = {"left": "left_base_link", "right": "right_base_link"}
    tcp_sites = ("left_tcp", "right_tcp")

    @cached_property
    def source_model(self):
        return mujoco.MjModel.from_xml_path(str(SOURCE))

    @cached_property
    def tcp_transform(self):
        model = self.source_model
        data = mujoco.MjData(model)
        data.qpos[model.joint("joint7").qposadr[0]] = .02
        data.qpos[model.joint("joint8").qposadr[0]] = -.02
        mujoco.mj_forward(model, data)
        tip = model.body("link6").id
        rotation = data.xmat[tip].reshape(3, 3)
        pads = [data.xpos[model.body(name).id] + data.xmat[model.body(name).id].reshape(3, 3) @ point for name, point in PADS]
        center = rotation.T @ ((pads[0] + pads[1]) * .5 - data.xpos[tip])
        grasp = rotation.T @ (pads[0] - pads[1])
        grasp /= np.linalg.norm(grasp)
        approach = np.array([0., 0., 1.])
        approach -= grasp * (approach @ grasp)
        approach /= np.linalg.norm(approach)
        normal = np.cross(grasp, approach)
        return center, np.column_stack((approach, normal, grasp))

    def scene_spec(self):
        tree = ET.parse(SOURCE).getroot()
        tree.find("compiler").set("meshdir", str(SOURCE.parent / "assets"))
        for tag in ("actuator", "equality", "contact", "keyframe"):
            item = tree.find(tag)
            if item is not None:
                tree.remove(item)
        world = tree.find("worldbody")
        source = world.find("body")
        world.remove(source)
        center, rotation = self.tcp_transform
        for side in ("left", "right"):
            body = copy.deepcopy(source)
            tool = next(b for b in body.iter("body") if b.get("name") == "link6")
            ET.SubElement(tool, "site", name="tcp", pos=_numbers(center),
                          quat=_numbers(Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]), size="0.005")
            for elem in body.iter():
                if elem.get("name"):
                    elem.set("name", side + "_" + elem.get("name"))
            world.append(body)
        ET.SubElement(world, "camera", name=self.camera)
        return mujoco.MjSpec.from_string(ET.tostring(tree, encoding="unicode"))

    def _urdf(self, side):
        model = self.source_model
        root = ET.Element("robot", name=side + "_piper")
        prefix = side + "_"
        ET.SubElement(root, "link", name=prefix + "base_link")
        for i in range(1, 7):
            body = model.body(f"link{i}")
            joint = model.joint(f"joint{i}")
            ET.SubElement(root, "link", name=prefix + body.name)
            node = ET.SubElement(root, "joint", name=prefix + joint.name, type="revolute")
            ET.SubElement(node, "parent", link=prefix + model.body(int(body.parentid[0])).name)
            ET.SubElement(node, "child", link=prefix + body.name)
            rpy = Rotation.from_quat(body.quat[[1, 2, 3, 0]]).as_euler("xyz")
            ET.SubElement(node, "origin", xyz=_numbers(body.pos), rpy=_numbers(rpy))
            ET.SubElement(node, "axis", xyz=_numbers(joint.axis))
            ET.SubElement(node, "limit", lower=str(joint.range[0]), upper=str(joint.range[1]), effort="100", velocity="5.5")
        center, rotation = self.tcp_transform
        ET.SubElement(root, "link", name=prefix + "tcp")
        node = ET.SubElement(root, "joint", name=prefix + "tool_fixed", type="fixed")
        ET.SubElement(node, "parent", link=prefix + "link6")
        ET.SubElement(node, "child", link=prefix + "tcp")
        # The tool frame is at an Euler singularity. SciPy's canonical third-angle
        # choice represents exactly the same rotation (covered by the FK test).
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Gimbal lock detected.*", category=UserWarning)
            tool_rpy = Rotation.from_matrix(rotation).as_euler("xyz")
        ET.SubElement(node, "origin", xyz=_numbers(center), rpy=_numbers(tool_rpy))
        return ET.tostring(root, encoding="utf-8")

    @cached_property
    def groups(self):
        result = []
        self._urdf_files = []
        for side_index, side in enumerate(("left", "right")):
            # Keep the fd alive for this adapter; no disk asset or cleanup job is needed.
            fd = os.memfd_create(side + "_piper.urdf", os.MFD_CLOEXEC)
            handle = os.fdopen(fd, "w+b")
            handle.write(self._urdf(side))
            handle.flush()
            self._urdf_files.append(handle)
            joints = tuple(f"{side}_joint{i}" for i in range(1, 7))
            tools = (side + "_tcp",)
            lower, upper = self.limits(joints)
            home = self.source_model.key("home").qpos[:6].copy()
            config = {"robot_cfg": {"kinematics": {
                "urdf_path": f"/proc/self/fd/{fd}", "asset_root_path": "", "base_link": self.roots[side],
                "tool_frames": list(tools), "collision_link_names": None, "collision_spheres": None,
                "cspace": {"joint_names": list(joints), "default_joint_position": home.tolist(),
                           "null_space_weight": [1.] * 6, "cspace_distance_weight": [1.] * 6,
                           "max_acceleration": 15., "max_jerk": 500.}}}}
            result.append(SolveGroup(side, (side_index,), joints, tools, config, lower, upper, home, .626, BOUNDS.copy(), batch_key="piper"))
        return tuple(result)

    def gripper_positions(self, width, valid):
        result = {}
        for i, side in enumerate(("left", "right")):
            opening = np.where(valid[:, i], np.clip(width[:, i] * .5, 0., .035), 0.)
            result[side + "_joint7"] = opening
            result[side + "_joint8"] = -opening
        return result

    def body_side(self, name):
        for side in ("left", "right"):
            if name.startswith(side + "_"):
                return side
        return None

    def is_gripper(self, model, geom_id):
        name = model.body(int(model.geom_bodyid[geom_id])).name
        return model.geom_group[geom_id] == 2 and name in ("left_link7", "left_link8", "right_link7", "right_link8")

    def hide(self, model):
        model.geom_rgba[model.geom_group == 3, 3] = 0.
        model.site_rgba[:, 3] = 0.

    def quality_model(self):
        # The source already provides collision primitives separate from visual meshes.
        return self.scene_spec().compile()
