"""Render the ARX arm from the action outputs alone. Composite pastes it later."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egowhale.step import BASE, GRIPPER, IK, PREFIX, RENDER, ROOT, SCALE, Step, piece_dir, pieces

_SCENE = ROOT / "assets" / "mujoco_arx_scene" / "scene.xml"
_HIDE = ("floor", "table", "camera", "workspace", "front_workspace", "base_link", "base_plus", "base_minus", "robot_front", "humanego")
_GRIPPER_GEOMS = ("left_link7", "left_link8", "right_link17", "right_link18")
ARM, GRIPPER_PART = 1, 2


class Render(Step):
    name = "render"
    needs = (IK, BASE, GRIPPER, PREFIX)
    makes = (RENDER,)

    def run(self, src: Path, dst: Path) -> None:
        spans = pieces(dst)
        for index in range(len(spans)):
            if not (piece_dir(dst, index, len(spans)) / RENDER).is_file():
                self.run_piece(src, dst, index)

    def run_piece(self, src: Path, dst: Path, index: int) -> None:
        """Render one piece. The approach leads the first piece only."""
        dst = Path(dst)
        source_h, source_w = _video_size(Path(src).with_suffix(".mp4"))
        height, width = int(source_h * SCALE), int(source_w * SCALE)
        with np.load(dst / IK) as data:
            qpos = np.asarray(data["qpos"])
        with np.load(dst / GRIPPER) as data:
            intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
        intrinsic = intrinsic.copy()
        intrinsic[:2] *= height / source_h
        base = np.asarray(json.loads((dst / BASE).read_text())["matrix"], dtype=np.float64)
        spans = pieces(dst)
        start, stop = spans[index]
        if index == 0:
            rows = np.concatenate((np.load(dst / PREFIX)["qpos"], qpos[1:stop]), axis=0)
        else:
            rows = qpos[start:stop]
        rgb, robot_mask, gripper_mask, depth = _render(rows, base, intrinsic, height, width)
        part = np.where(gripper_mask, GRIPPER_PART, np.where(robot_mask, ARM, 0)).astype(np.uint8)
        rgb[~robot_mask] = 0
        depth[~gripper_mask] = 0
        path = piece_dir(dst, index, len(spans)) / RENDER
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rgb=rgb, part=part, depth=depth)


def load_render(path: Path):
    """rgb, robot mask, gripper mask, depth. rgb is kept on the robot, depth on the gripper."""
    with np.load(path) as data:
        part = data["part"]
        return data["rgb"], part > 0, part == GRIPPER_PART, data["depth"]


def _video_size(path: Path) -> tuple[float, float]:
    capture = cv2.VideoCapture(str(path))
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    capture.release()
    if height <= 0 or width <= 0:
        raise FileNotFoundError(path)
    return height, width


_SCENES: dict[tuple[int, int], tuple] = {}


def _scene(height: int, width: int):
    """One model and GL context per image size, kept for the life of the process."""
    if (height, width) not in _SCENES:
        model = mujoco.MjModel.from_xml_path(str(_SCENE))
        _hide(model)
        gripper_ids = np.array([
            geom_id
            for geom_id in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").startswith(_GRIPPER_GEOMS)
        ])
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        renderer = mujoco.Renderer(model, height=height, width=width)
        _SCENES[(height, width)] = (model, mujoco.MjData(model), renderer, gripper_ids)
    return _SCENES[(height, width)]


def _render(qpos, base, intrinsic, height, width):
    model, data, renderer, gripper_ids = _scene(height, width)
    _place_base(model, base)
    _place_camera(model, intrinsic, height)
    count = len(qpos)
    rgb = np.zeros((count, height, width, 3), dtype=np.uint8)
    robot_mask = np.zeros((count, height, width), dtype=bool)
    gripper_mask = np.zeros((count, height, width), dtype=bool)
    depth = np.zeros((count, height, width), dtype=np.float32)
    for index in range(count):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="ego_camera_calibrated")
        rgb[index] = renderer.render()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="ego_camera_calibrated")
        geom = renderer.render()[:, :, 0]
        renderer.disable_segmentation_rendering()
        robot_mask[index] = geom >= 0
        gripper_mask[index] = np.isin(geom, gripper_ids)
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="ego_camera_calibrated")
        depth[index] = renderer.render()
        renderer.disable_depth_rendering()
    return rgb, robot_mask, gripper_mask, depth


def _hide(model: mujoco.MjModel) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith(_HIDE):
            model.geom_rgba[geom_id, 3] = 0.0
    model.site_rgba[:, 3] = 0.0


def _place_base(model: mujoco.MjModel, transform: np.ndarray) -> None:
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    model.body_pos[body] = transform[:3, 3]
    model.body_quat[body] = Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]]


def _place_camera(model: mujoco.MjModel, intrinsic: np.ndarray, height: int) -> None:
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ego_camera_calibrated")
    model.cam_pos[cam] = 0.0
    model.cam_quat[cam] = Rotation.from_matrix(np.diag([1.0, -1.0, -1.0])).as_quat()[[3, 0, 1, 2]]
    model.cam_fovy[cam] = float(np.degrees(2.0 * np.arctan(height / (2.0 * float(intrinsic[1, 1])))))
