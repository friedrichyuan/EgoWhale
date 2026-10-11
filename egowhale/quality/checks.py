"""Level 1: per-frame checks of the pipeline's own output, from the arm rendered in a collision-aware scene.

A frame passes when both hands were tracked, IK reached them, the arm is in view without filling it, and the
arm neither hits itself nor crosses over to the other arm.
"""

from __future__ import annotations

import mujoco
import numpy as np

from egowhale.robot import CAMERA, SCENE, TCP, hide, place_base, place_camera

# Mean distance from the IK tool points to the tracked grippers, in meters.
IK_LIMIT = 0.05
# Share of the image the arm may cover.
COVERAGE_LIMIT = 0.70
# Shortest run of passing frames, in seconds, kept between failing ones.
MIN_RUN_SECONDS = 0.3


def _load_model():
    """Visual meshes stay hidden from collision. Convex copies are what contacts use."""
    spec = mujoco.MjSpec.from_file(str(SCENE))
    for body in spec.bodies:
        if body.name != "base_link" and not body.name.startswith(("left_link", "right_link")):
            continue
        for geom in list(body.geoms):
            if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            body.add_geom(
                name=f"{geom.name}_hit",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=geom.meshname,
                pos=geom.pos,
                quat=geom.quat,
                contype=1,
                conaffinity=1,
                group=3,
                rgba=[0.0, 0.0, 0.0, 0.0],
            )
    return spec.compile()


_SCENES: dict[tuple[int, int], tuple] = {}


def _scene(height: int, width: int):
    """One model and GL context per image size, kept for the life of the process."""
    if (height, width) not in _SCENES:
        model = _load_model()
        hide(model)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        renderer = mujoco.Renderer(model, height=height, width=width)
        _SCENES[(height, width)] = (model, mujoco.MjData(model), renderer)
    return _SCENES[(height, width)]


def frame_checks(qpos, base, intrinsic, position, hand_valid, height, width) -> dict[str, np.ndarray]:
    """Per-frame signals for the arm at `qpos`, seen from the video camera at height x width."""
    model, data, renderer = _scene(height, width)
    place_base(model, base)
    place_camera(model, intrinsic, height)
    bodies = _robot_bodies(model)
    parent = np.asarray(model.body_parentid)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in TCP]
    count = len(qpos)
    area = float(height * width)
    ik_error = np.full(count, np.nan)
    visible = np.zeros(count, dtype=bool)
    coverage = np.zeros(count, dtype=bool)
    self_collision = np.zeros(count, dtype=bool)
    cross_arm = np.zeros(count, dtype=bool)
    for index in range(count):
        data.qpos[:] = qpos[index]
        mujoco.mj_forward(model, data)
        errors = []
        for side, site in enumerate(sites):
            if hand_valid[index, side]:
                errors.append(float(np.linalg.norm(data.site_xpos[site] - position[index, side])))
        if errors:
            ik_error[index] = float(np.mean(errors))
        same, cross = _contacts(model, data, bodies, parent)
        self_collision[index] = same > 0
        cross_arm[index] = cross > 1
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera=CAMERA)
        pixels = int(np.count_nonzero(renderer.render()[:, :, 0] >= 0))
        renderer.disable_segmentation_rendering()
        visible[index] = pixels > 0
        coverage[index] = pixels / area > COVERAGE_LIMIT
    return {
        "hand": hand_valid.all(axis=1),
        "ik": np.isfinite(ik_error) & (ik_error < IK_LIMIT),
        "ik_error": ik_error,
        "visible": visible,
        "coverage": coverage,
        "self_collision": self_collision,
        "cross_arm": cross_arm,
    }


def passed(signals) -> np.ndarray:
    """Frames where every level-1 check holds."""
    return (
        signals["hand"]
        & signals["ik"]
        & signals["visible"]
        & ~signals["self_collision"]
        & ~signals["cross_arm"]
        & ~signals["coverage"]
    )


def erode(valid: np.ndarray, fps: float) -> np.ndarray:
    """Drop passing runs shorter than MIN_RUN_SECONDS that sit between failing frames."""
    limit = int(np.floor(MIN_RUN_SECONDS * fps))
    out = valid.copy()
    start = 0
    while start < len(valid):
        if not valid[start]:
            start += 1
            continue
        stop = start + 1
        while stop < len(valid) and valid[stop]:
            stop += 1
        sandwiched = start > 0 and stop < len(valid)
        if sandwiched and stop - start < limit:
            out[start:stop] = False
        start = stop
    return out


def _robot_bodies(model) -> dict[int, str]:
    bodies = {}
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if name == "base_link" or name.startswith(("left_link", "right_link")):
            bodies[body_id] = name
    return bodies


def _near(parent, first: int, second: int) -> bool:
    """Parent and grandparent hulls overlap on a straight arm. Those are not collisions."""
    depth = {}
    node, hops = first, 0
    while node > 0 and hops < 40:
        depth[node] = hops
        node = int(parent[node])
        hops += 1
    node, hops = second, 0
    while node > 0 and hops < 40:
        if node in depth:
            return hops + depth[node] < 3
        node = int(parent[node])
        hops += 1
    return False


def _contacts(model, data, bodies, parent):
    same = cross = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        first = int(model.geom_bodyid[contact.geom1])
        second = int(model.geom_bodyid[contact.geom2])
        if first not in bodies or second not in bodies or first == second or _near(parent, first, second):
            continue
        first_left = bodies[first].startswith("left_")
        second_left = bodies[second].startswith("left_")
        if first_left != second_left and "base_link" not in (bodies[first], bodies[second]):
            cross += 1
        else:
            same += 1
    return same, cross
