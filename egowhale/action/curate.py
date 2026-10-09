"""Quality curation: pipeline checks, action statistics, then a VLM audit."""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from egowhale.media import read_frames, video_shape
from egowhale.step import BASE, COMPOSITE, GRIPPER, IK, PREFIX, QUALITY, Step
from egowhale.visual.render import _hide, _place_base, _place_camera

_SCENE = Path(__file__).resolve().parents[2] / "assets" / "mujoco_arx_scene" / "scene.xml"
_TCP = ("left_tcp", "right_tcp")
_IK_LIMIT = 0.05
_COVERAGE_LIMIT = 0.70
_DROP_RATIO = 0.60
_PROMPT = """You are evaluating whether a manipulation video matches its text description.
This is a ROBOT manipulation dataset. "Hand" refers to the robot's gripper/end-effector.
Many tasks use FAKE or SIMULATED objects as stand-ins for real objects. This is EXPECTED.
Task Description: {description}
Determine if the robot's actions match the described task.
Flag MAJOR MISMATCHES: wrong action type, wrong object category, wrong target location, or failed execution.
Be tolerant of: fake/toy objects, minor appearance variations, small spatial deviations, different grasping approaches.
Respond in JSON: {"is consistent": true/false, "confidence": 0.0-1.0, "reasoning": "..."}
"""


class Curate(Step):
    name = "curate"
    needs = (GRIPPER, IK, BASE, COMPOSITE, PREFIX)
    makes = (QUALITY,)

    def run(self, src: Path, dst: Path) -> None:
        dst = Path(dst)
        with np.load(dst / GRIPPER) as data:
            position = np.asarray(data["position"], dtype=np.float64)
            rotation = np.asarray(data["rotation"], dtype=np.float64)
            width = np.asarray(data["width"], dtype=np.float64)
            valid = np.asarray(data["valid"], dtype=bool)
            fps = float(data["fps"])
            intrinsic = np.asarray(data["intrinsic"], dtype=np.float64).copy()
        with np.load(dst / IK) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
        base = np.asarray(json.loads((dst / BASE).read_text())["matrix"], dtype=np.float64)
        frames = min(len(position), len(qpos))
        position, rotation, width, valid, qpos = (
            position[:frames],
            rotation[:frames],
            width[:frames],
            valid[:frames],
            qpos[:frames],
        )
        _count, video_h, video_w, _video_fps = video_shape(dst / COMPOSITE)
        scale = video_h / _source_height(Path(src).with_suffix(".mp4"))
        intrinsic[:2] *= scale
        signals = _signals(qpos, base, intrinsic, position, valid, video_h, video_w)
        l1 = _erode(_l1_mask(signals), fps)
        state, action = _trajectory(position, rotation, width)
        outlier = _outliers(action, l1)
        sudden = _sudden(state, l1)
        sudden[1:] |= _sudden(action, l1[:-1] & l1[1:])
        kept = l1 & ~outlier & ~sudden
        invalid_ratio = float((~kept).mean())
        keep = invalid_ratio <= _DROP_RATIO
        description = _description(Path(src))
        approach = len(np.load(dst / PREFIX)["qpos"]) - 1
        audit = _audit(dst / COMPOSITE, approach, frames, fps, description) if keep else {"status": "skipped", "reason": "dropped before audit"}
        if audit.get("is_consistent") is False:
            keep = False
        report = {
            "keep": keep,
            "description": description,
            "invalid_ratio": invalid_ratio,
            "mean_ik_error_m": _finite_mean(signals["ik_error"]),
            "hand": _bits(signals["hand"]),
            "ik": _bits(signals["ik"]),
            "visible": _bits(signals["visible"]),
            "self_collision": _bits(signals["self_collision"]),
            "cross_arm": _bits(signals["cross_arm"]),
            "too_large": _bits(signals["coverage"]),
            "l1": _bits(l1),
            "outlier": _bits(outlier),
            "sudden": _bits(sudden),
            "valid": _bits(kept),
            "vlm": audit,
        }
        path = dst / QUALITY
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"  keep {int(kept.sum())}/{frames}  invalid {invalid_ratio:.1%}  vlm {audit['status']}")


def _load_model():
    """Visual meshes stay hidden from collision. Convex copies are what contacts use."""
    spec = mujoco.MjSpec.from_file(str(_SCENE))
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
        _hide(model)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        renderer = mujoco.Renderer(model, height=height, width=width)
        _SCENES[(height, width)] = (model, mujoco.MjData(model), renderer)
    return _SCENES[(height, width)]


def _signals(qpos, base, intrinsic, position, hand_valid, height, width):
    model, data, renderer = _scene(height, width)
    _place_base(model, base)
    _place_camera(model, intrinsic, height)
    bodies = _robot_bodies(model)
    parent = np.asarray(model.body_parentid)
    sites = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) for name in _TCP]
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
        renderer.update_scene(data, camera="ego_camera_calibrated")
        pixels = int(np.count_nonzero(renderer.render()[:, :, 0] >= 0))
        renderer.disable_segmentation_rendering()
        visible[index] = pixels > 0
        coverage[index] = pixels / area > _COVERAGE_LIMIT
    return {
        "hand": hand_valid.all(axis=1),
        "ik": np.isfinite(ik_error) & (ik_error < _IK_LIMIT),
        "ik_error": ik_error,
        "visible": visible,
        "coverage": coverage,
        "self_collision": self_collision,
        "cross_arm": cross_arm,
    }


def _l1_mask(signals) -> np.ndarray:
    return (
        signals["hand"]
        & signals["ik"]
        & signals["visible"]
        & ~signals["self_collision"]
        & ~signals["cross_arm"]
        & ~signals["coverage"]
    )


def _erode(valid: np.ndarray, fps: float) -> np.ndarray:
    limit = int(np.floor(0.3 * fps))
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


def _trajectory(position, rotation, width):
    state = np.concatenate((position.reshape(len(position), -1), width), axis=1)
    pos_delta = np.diff(position.reshape(len(position), -1), axis=0)
    width_delta = np.diff(width, axis=0)
    rot_delta = np.full((len(position) - 1, 6), np.nan, dtype=np.float64)
    for side in range(2):
        good = np.isfinite(rotation[:-1, side]).all(axis=(1, 2)) & np.isfinite(rotation[1:, side]).all(axis=(1, 2))
        if not good.any():
            continue
        rot_delta[good, side * 3 : (side + 1) * 3] = (
            Rotation.from_matrix(rotation[:-1, side][good]).inv() * Rotation.from_matrix(rotation[1:, side][good])
        ).as_rotvec()
    action = np.concatenate((pos_delta, rot_delta, width_delta), axis=1)
    return state, action


def _outliers(action: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flagged = np.zeros(len(valid), dtype=bool)
    usable = valid[:-1] & valid[1:] & np.isfinite(action).all(axis=1)
    if int(usable.sum()) < 4:
        return flagged
    low, high = _band(action[usable])
    outside = usable & np.any((action < low) | (action > high), axis=1)
    flagged[1:] = outside
    return flagged


def _sudden(series: np.ndarray, valid: np.ndarray) -> np.ndarray:
    flagged = np.zeros(len(series), dtype=bool)
    kernels = (
        np.array([-1.0, 1.0]),
        np.array([1.0, -2.0, 1.0]),
        np.array([-1.0, 3.0, -3.0, 1.0]),
    )
    finite = valid & np.isfinite(series).all(axis=1)
    for kernel in kernels:
        width = len(kernel)
        rows, ends = [], []
        for end in range(width - 1, len(series)):
            if not finite[end - width + 1 : end + 1].all():
                continue
            rows.append(series[end - width + 1 : end + 1].T @ kernel)
            ends.append(end)
        if len(rows) < 4:
            continue
        values = np.abs(np.stack(rows))
        limit = _upper(values)
        over = np.any(values > limit, axis=1)
        for end, hit in zip(ends, over):
            flagged[end] |= bool(hit)
    return flagged


def _band(values: np.ndarray):
    low = np.quantile(values, 0.01, axis=0)
    high = np.quantile(values, 0.99, axis=0)
    span = high - low
    return low - 3.0 * span, high + 3.0 * span


def _upper(values: np.ndarray) -> np.ndarray:
    _low, high = _band(values)
    return high


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


def _audit(video: Path, start: int, frames: int, fps: float, description: str) -> dict:
    """Ask the VLM about `frames` frames from `start`, sampled at 4 fps. Only those frames are decoded."""
    key = os.environ.get("EGOWHALE_VLM_API_KEY", "").strip()
    if not key:
        return {"status": "skipped", "reason": "EGOWHALE_VLM_API_KEY is unset"}
    count = min(frames, video_shape(video)[0] - start)
    step = max(fps / 4.0, 1.0)
    indices = np.unique(np.round(np.arange(0, count, step)).astype(int))
    indices = indices[indices < count]
    content = []
    for frame in read_frames(video, start + indices) if len(indices) else []:
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            continue
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{payload}"}})
    content.append({"type": "text", "text": _PROMPT.replace("{description}", description)})
    body = json.dumps({
        "model": os.environ.get("EGOWHALE_VLM_MODEL", "qwen3.8-flash"),
        "messages": [{"role": "user", "content": content}],
    }).encode("utf-8")
    base = os.environ.get("EGOWHALE_VLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/")
    request = urllib.request.Request(
        f"{base}/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        reply = json.loads(response.read().decode("utf-8"))
    text = reply["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
    return _verdict(str(text))


def _verdict(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match is None:
        return {"status": "error", "raw": text}
    payload = json.loads(match.group(0))
    consistent = payload.get("is consistent", payload.get("is_consistent"))
    return {
        "status": "ok",
        "is_consistent": None if consistent is None else bool(consistent),
        "confidence": payload.get("confidence"),
        "reasoning": payload.get("reasoning"),
    }


def _description(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        for key in ("llm_description", "task"):
            if key not in handle.attrs:
                continue
            value = handle.attrs[key]
            if isinstance(value, bytes):
                value = value.decode()
            text = str(value).strip()
            if text and text.lower() != "none":
                return text
    return "manipulation"


def _source_height(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)
    capture.release()
    if height <= 0:
        raise FileNotFoundError(path)
    return height


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(finite.mean())


def _bits(values: np.ndarray) -> list[int]:
    return [int(bool(value)) for value in values]
