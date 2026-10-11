"""Paste the rendered ARX arm onto the inpainted video. The arm is pasted; the gripper uses depth."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from egowhale.media import RgbWriter, load_masks, read_rgb
from egowhale.paths import COMPOSITE, DEPTH, INPAINT, MASKS, PREFIX, RENDER
from egowhale.pieces import piece_dir, pieces
from egowhale.step import Step
from egowhale.visual.render import load_render

_HAND_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class Composite(Step):
    name = "composite"
    needs = (INPAINT, RENDER, DEPTH, MASKS, PREFIX)
    makes = (COMPOSITE,)

    def run(self, src: Path, dst: Path) -> None:
        """Pieces are merged here: each is composited in order and appended to one video."""
        dst = Path(dst)
        prefix = len(np.load(dst / PREFIX)["qpos"])
        spans = pieces(dst)
        writer = None
        try:
            for index in range(len(spans)):
                where = piece_dir(dst, index, len(spans))
                background, fps = read_rgb(where / INPAINT)
                robot, robot_mask, gripper_mask, robot_depth = load_render(where / RENDER)
                # The first piece's render opens with the approach, held on its first frame.
                lead = prefix if index == 0 else 1
                frames = min(len(background), len(robot) - lead + 1)
                background = _hold(background[:frames], lead)
                height, width = background.shape[1], background.shape[2]
                scene_depth = _hold(_match_depth(np.load(where / DEPTH)["depth"][:frames], height, width), lead)
                hand = _hold(_match_mask(load_masks(where / MASKS)[:frames], height, width), lead)
                if writer is None:
                    writer = RgbWriter(dst / COMPOSITE, fps)
                writer.write(_composite(background, robot, robot_mask, gripper_mask, robot_depth, scene_depth, hand))
        finally:
            if writer is not None:
                writer.close()


def _hold(array: np.ndarray, count: int) -> np.ndarray:
    """Keep frame 0 under the approach, then continue from frame 1. Frame 0 is the arrival."""
    if count == 1:
        return array
    return np.concatenate((np.repeat(array[:1], count, axis=0), array[1:]), axis=0)


def _composite(background, robot, robot_mask, gripper_mask, robot_depth, scene_depth, hand):
    count = min(len(background), len(robot), len(scene_depth), len(hand))
    image = background[:count].copy()
    for index in range(count):
        dilated = cv2.dilate(hand[index].astype(np.uint8), _HAND_KERNEL, iterations=1).astype(bool)
        arm = robot_mask[index] & ~gripper_mask[index]
        hidden = gripper_mask[index] & (scene_depth[index] < robot_depth[index]) & ~dilated
        visible = arm | (gripper_mask[index] & ~hidden)
        image[index][visible] = robot[index][visible]
    return image


def _match_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    if depth.shape[1:] == (height, width):
        return depth
    return np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth])


def _match_mask(masks: np.ndarray, height: int, width: int) -> np.ndarray:
    if masks.shape[1:] == (height, width):
        return masks
    return np.stack([
        cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        for mask in masks
    ])
