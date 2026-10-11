"""The ARX dual arm: asset paths, joint names, and the MuJoCo scene helpers render and curate share."""

from __future__ import annotations

import functools
import os
import tempfile
from pathlib import Path

import numpy as np

from egowhale.paths import ASSETS

SCENE = ASSETS / "mujoco_arx_scene" / "scene.xml"
URDF = ASSETS / "curobo_arx" / "arx_acone_kin.urdf"
YAML = ASSETS / "curobo_arx" / "arx_acone.yml"

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
CAMERA = "ego_camera_calibrated"
# Scene geoms that are not the arm. Render and curate draw only the robot over the video.
_HIDE = ("floor", "table", "camera", "workspace", "front_workspace", "base_link", "base_plus", "base_minus", "robot_front", "humanego")


@functools.cache
def robot_yaml() -> str:
    """cuRobo's robot file with absolute asset paths, written once per process.

    One file per process: actors on one host would otherwise rewrite each other's file while cuRobo reads it.
    """
    text = YAML.read_text(encoding="utf-8")
    text = text.replace('urdf_path: "arx_acone_kin.urdf"', f'urdf_path: "{URDF}"')
    text = text.replace('asset_root_path: ""', f'asset_root_path: "{URDF.parent}"')
    path = Path(tempfile.gettempdir()) / f"egowhale_arx_{os.getpid()}.yml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def hide(model) -> None:
    """Make everything but the arm transparent, and every site."""
    import mujoco

    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(_HIDE):
            model.geom_rgba[geom_id, 3] = 0.0
    model.site_rgba[:, 3] = 0.0


def place_base(model, transform: np.ndarray) -> None:
    """Put the robot base at a 4x4 camera-frame pose."""
    import mujoco
    from scipy.spatial.transform import Rotation

    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    model.body_pos[body] = transform[:3, 3]
    model.body_quat[body] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]


def place_camera(model, intrinsic: np.ndarray, height: int) -> None:
    """The scene camera at the origin, looking down +z like OpenCV, with the video's vertical field of view."""
    import mujoco
    from scipy.spatial.transform import Rotation

    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA)
    model.cam_pos[cam] = 0.0
    model.cam_quat[cam] = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0])).as_quat()[[3, 0, 1, 2]]
    model.cam_fovy[cam] = float(np.degrees(2.0 * np.arctan(height / (2.0 * float(intrinsic[1, 1])))))
