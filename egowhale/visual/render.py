"""Render the ARX arm from the action outputs alone. Composite pastes it later."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from egowhale.media import SCALE, frame_size
from egowhale.paths import BASE, GRIPPER, IK, PREFIX, RENDER
from egowhale.robots import load_robot
from egowhale.robots.motion import read_bases, read_motion
from egowhale.step import Step

GRIPPER_PART = 2


class Render(Step):
    name = "render"
    needs = (IK, BASE, GRIPPER)
    makes = (RENDER,)
    per_piece = True

    def __init__(self, robot="arx"):
        super().__init__()
        self.robot = load_robot(robot)

    def run_piece(self, src: Path, dst: Path, index: int) -> None:
        """Render one piece. The approach leads the first piece only."""
        dst = Path(dst)
        source_h, source_w = frame_size(Path(src).with_suffix(".mp4"))
        height, width = int(source_h * SCALE), int(source_w * SCALE)
        qpos = self.robot.qpos(read_motion(dst / IK))
        with np.load(dst / GRIPPER) as data:
            intrinsic = np.asarray(data["intrinsic"], dtype=np.float64)
        intrinsic = intrinsic.copy()
        intrinsic[:2] *= height / source_h
        base = read_bases(dst / BASE)
        where, start, stop = self.piece(dst, index)
        if index == 0 and (dst / PREFIX).is_file():
            rows = np.concatenate((self.robot.qpos(read_motion(dst / PREFIX)), qpos[1:stop]), axis=0)
        else:
            rows = qpos[start:stop]
        rgb, robot_mask, gripper_mask, depth = _render(rows, base, intrinsic, height, width, self.robot)
        part = _postprocess(rgb, robot_mask, gripper_mask, depth)
        path = where / RENDER
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rgb=rgb, part=part, depth=depth)


def _postprocess(rgb: np.ndarray, robot_mask: np.ndarray, gripper_mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Keep the same pixels while avoiding full-frame int64 temporaries and boolean RGB indexing."""
    part = robot_mask.astype(np.uint8)
    np.copyto(part, 2, where=gripper_mask)
    np.multiply(rgb, robot_mask[..., None], out=rgb)
    depth[~gripper_mask] = 0
    return part


def load_render(path: Path):
    """rgb, robot mask, gripper mask, depth. rgb is kept on the robot, depth on the gripper."""
    with np.load(path) as data:
        part = data["part"]
        return data["rgb"], part > 0, part == GRIPPER_PART, data["depth"]


def _scene(height: int, width: int, robot):
    """One model and GL context per image size, kept for the life of the process."""
    scenes = robot.__dict__.setdefault("_render_scenes", {})
    if (height, width) not in scenes:
        model = robot.scene_spec().compile()
        robot.hide(model)
        gripper_ids = np.array([
            geom_id
            for geom_id in range(model.ngeom)
            if robot.is_gripper(model, geom_id)
        ])
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        renderer = mujoco.Renderer(model, height=height, width=width)
        scenes[(height, width)] = (model, mujoco.MjData(model), renderer, gripper_ids)
    return scenes[(height, width)]


def _render(qpos, base, intrinsic, height, width, robot=None):
    robot = load_robot() if robot is None else robot
    model, data, renderer, gripper_ids = _scene(height, width, robot)
    robot.place_bases(model, base)
    robot.place_camera(model, intrinsic, height)
    count = len(qpos)
    rgb = np.zeros((count, height, width, 3), dtype=np.uint8)
    robot_mask = np.zeros((count, height, width), dtype=bool)
    gripper_mask = np.zeros((count, height, width), dtype=bool)
    depth = np.zeros((count, height, width), dtype=np.float32)
    for index in range(count):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=robot.camera)
        rgb[index] = renderer.render()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=robot.camera)
        geom = renderer.render()[:, :, 0]
        renderer.disable_segmentation_rendering()
        robot_mask[index] = geom >= 0
        gripper_mask[index] = np.isin(geom, gripper_ids)
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=robot.camera)
        depth[index] = renderer.render()
        renderer.disable_depth_rendering()
    return rgb, robot_mask, gripper_mask, depth
