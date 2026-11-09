"""ARX adapter preserving the original asset conventions and search settings."""
from functools import cached_property

import mujoco
import numpy as np
import yaml

from egowhale.paths import ASSETS

from .base import Robot, SolveGroup

ARM = tuple([f"left_joint{i}" for i in range(1, 7)] + [f"right_joint{i}" for i in range(11, 17)])
GRIPPERS = (("left_joint7", "left_joint8"), ("right_joint17", "right_joint18"))


class ARXRobot(Robot):
    name = "arx"
    roots = {"base": "base_link"}
    tcp_sites = ("left_tcp", "right_tcp")

    def scene_spec(self):
        return mujoco.MjSpec.from_file(str(ASSETS / "mujoco_arx_scene/scene.xml"))

    @cached_property
    def groups(self):
        path = ASSETS / "curobo_arx/arx_acone.yml"
        config = yaml.safe_load(path.read_text())
        kin = config["robot_cfg"]["kinematics"]
        kin["urdf_path"] = str((path.parent / kin["urdf_path"]).resolve())
        kin["asset_root_path"] = str(path.parent.resolve())
        lower, upper = self.limits(ARM)
        return (SolveGroup("base", (0, 1), ARM, self.tcp_sites, config, lower, upper,
                           np.zeros(len(ARM)), .855, np.array([.30, .30, .20, np.deg2rad(35.)])),)

    def gripper_positions(self, width, valid):
        model = self.model
        data = mujoco.MjData(model)
        values = {}
        for side, names in enumerate(GRIPPERS):
            joints = [model.joint(n) for n in names]
            data.qpos[:] = 0.
            for joint in joints:
                data.qpos[joint.qposadr[0]] = joint.range[0]
            mujoco.mj_forward(model, data)
            gap = float(np.linalg.norm(data.xpos[joints[0].bodyid[0]] - data.xpos[joints[1].bodyid[0]]))
            upper = min(float(j.range[1]) for j in joints)
            opening = np.where(valid[:, side], np.clip(.5 * (width[:, side] - gap), 0., upper), 0.)
            values.update({name: opening for name in names})
        return values

    def body_side(self, name):
        if name == "base_link":
            return "shared"
        if name.startswith("left_link"):
            return "left"
        if name.startswith("right_link"):
            return "right"
        return None

    def is_gripper(self, model, geom_id):
        return (model.geom(geom_id).name or "").startswith(("left_link7", "left_link8", "right_link17", "right_link18"))

    def hide(self, model):
        prefixes = ("floor", "table", "camera", "workspace", "front_workspace", "base_link", "base_plus", "base_minus", "robot_front", "humanego")
        for i in range(model.ngeom):
            if (model.geom(i).name or "").startswith(prefixes):
                model.geom_rgba[i, 3] = 0.
        model.site_rgba[:, 3] = 0.
