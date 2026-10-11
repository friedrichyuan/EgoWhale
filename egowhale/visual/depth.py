"""Scene depth from Depth Anything 3 in short independent windows, scaled to meters by the tracked hand."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import h5py
import numpy as np

from egowhale import thirdparty
from egowhale.media import SCALE, frame_size, read_rgb, save_depth
from egowhale.paths import DEPTH, THIRDPARTY
from egowhale.step import Result, Step, failed_items, run_grouped

_WEIGHTS = THIRDPARTY / "da3" / "weights" / "DA3-GIANT"
PROCESS_RES = 336
# Views per DA3 window. Depth only orders the gripper against the hand and objects in the same frame,
# so frames need no shared context. On a 4090, 1-view windows ran at 34-35 fps, 4-view at 32-33, 64-view at 17.
WINDOW = 1
# Frames per forward. Short windows take about 0.19 GiB per frame; 96 leaves room for a cuRobo actor.
BUDGET = 96
_HAND_PARTS = ("Hand", "Finger", "Thumb")
# Joints a window needs to take its scale from the hand. Both hands in view give 50 per frame.
_HAND_MIN = 10


class Depth(Step):
    name = "depth"
    makes = (DEPTH,)
    gpus = 1
    per_piece = True

    def __init__(self, window: int = WINDOW) -> None:
        super().__init__()
        self.window = window
        self.model = None

    def load(self) -> None:
        if self.model is None:
            self.model = load_model()

    def release(self) -> None:
        self.model = None

    def prepare(self, src: Path, where: Path, start: int, stop: int) -> dict:
        """The piece resized for DA3, with its cameras and hand joints, for a CPU reader. Reads the source video."""
        frames, _fps = read_rgb(Path(src).with_suffix(".mp4"), start, stop)
        view = prepare_view(Path(src), frames, start)
        view["dst"] = str(Path(where) / DEPTH)
        return view

    def run_piece(self, src: Path, dst: Path, index: int) -> None:
        item = self.prepare(src, *self.piece(dst, index))
        item["ok"] = True
        results, saves, _limit = self.estimate([item], BUDGET)
        if not results[0].ran:
            raise RuntimeError(results[0].error)
        self.store(saves[0])
        print(f"  depth median {float(np.median(saves[0]['depth'])):.3f} m")

    def consume(self, payload: dict, budget: int | None = None) -> dict:
        import ray

        results, saves, limit = self.estimate(payload["items"], budget or BUDGET)
        return {"results": results, "budget": limit, "save_ref": ray.put(saves) if saves else None}

    def store(self, save: dict) -> None:
        save_depth(Path(save["dst"]), save["depth"])

    def estimate(self, items: list[dict], budget: int) -> tuple[list, list, int]:
        """Cut every piece into windows, forward equal windows together, join each piece back. OOM halves the budget."""
        results = failed_items(items)
        views = [(index, view) for index, item in enumerate(items) if item.get("ok") for view in _windows(item, self.window)]
        made = [None] * len(views)
        self.load()

        def forward(group: list[int]) -> None:
            produced = _forward_ready([views[position][1] for position in group], self.model)
            for position, depth in zip(group, produced):
                made[position] = depth

        started = time.perf_counter()
        sizes = {position: len(view["images"]) for position, (_owner, view) in enumerate(views)}
        with self.lock:
            errors, limit = run_grouped(list(range(len(views))), sizes, budget, forward)
        for position, message in errors.items():
            results[views[position][0]] = Result(False, 0.0, message)
        frames = sum(len(item["images"]) for item in items if item.get("ok"))
        share = (time.perf_counter() - started) / max(frames, 1)
        saves = []
        for index, item in enumerate(items):
            if results[index] is not None:
                continue
            mine = [(made[position], view["keep"]) for position, (owner, view) in enumerate(views) if owner == index]
            # A window with too few joints in view borrows the piece's median hand scale, then the camera path.
            found = [scale for (_depth, scale, _camera), _keep in mine if scale is not None]
            borrowed = float(np.median(found)) if found else None
            parts = [
                depth[keep:] * (scale if scale is not None else borrowed if borrowed is not None else camera)
                for (depth, scale, camera), keep in mine
            ]
            results[index] = Result(True, share * len(item["images"]))
            saves.append({"stage": self.name, "dst": item["dst"], "depth": np.concatenate(parts).astype(np.float32)})
        return results, saves, limit


def _windows(item: dict, window: int) -> list[dict]:
    """Consecutive windows of `window` views. The last is moved back to stay full; `keep` skips the views it repeats."""
    count = len(item["images"])
    size = min(window, count)
    starts = list(range(0, count - size + 1, size))
    if starts[-1] + size < count:
        starts.append(count - size)
    hand = item["hand"]
    views = []
    covered = 0
    for start in starts:
        stop = start + size
        rows = hand[(hand[:, 0] >= start) & (hand[:, 0] < stop)].copy()
        rows[:, 0] -= start
        views.append({
            "images": item["images"][start:stop],
            "extrinsics": item["extrinsics"][start:stop],
            "intrinsics": item["intrinsics"][start:stop],
            "size": item["size"],
            "hand": rows,
            "key": (size, *item["images"].shape[1:]),
            "keep": covered - start,
        })
        covered = stop
    return views


def _cameras(episode: Path, height: int, width: int, start: int, frames: int):
    with h5py.File(episode, "r") as handle:
        c2w = np.asarray(handle["transforms/camera"][start : start + frames], dtype=np.float64)
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
    source_h, source_w = frame_size(episode.with_suffix(".mp4"))
    intrinsic = intrinsic.copy()
    intrinsic[0] *= width / source_w
    intrinsic[1] *= height / source_h
    extrinsics = np.stack([_invert(camera) for camera in c2w])
    return extrinsics, np.repeat(intrinsic[None], len(c2w), axis=0)


def load_model():
    """DA3-GIANT kept on a depth actor. Serial runs load and drop it per episode."""
    if not (_WEIGHTS / "model.safetensors").is_file():
        raise FileNotFoundError(_WEIGHTS)
    thirdparty.use_da3()
    from depth_anything_3.cfg import create_object, load_config
    from depth_anything_3.registry import MODEL_REGISTRY
    from safetensors.torch import load_file

    model = create_object(load_config(MODEL_REGISTRY["da3-giant"])).cuda().eval()
    state = {key.removeprefix("model."): value for key, value in load_file(str(_WEIGHTS / "model.safetensors")).items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    depth_missing = [key for key in missing if "output_conv2." in key and "aux" not in key]
    if unexpected or depth_missing:
        raise RuntimeError(f"DA3 weight mismatch, missing {depth_missing[:5]}, unexpected {list(unexpected)[:5]}")
    return model


def _out_size(frames: np.ndarray) -> tuple[int, int]:
    """Depth is stored at the inpainted background's size."""
    return int(frames.shape[1] * SCALE), int(frames.shape[2] * SCALE)


def _hand_points(episode: Path, start: int, count: int) -> np.ndarray:
    """EgoDex hand joints as (frame from `start`, row, col, depth in meters), in stored-depth pixels."""
    with h5py.File(episode, "r") as handle:
        transforms = handle["transforms"]
        c2w = np.asarray(transforms["camera"][start : start + count], dtype=np.float64)
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        names = [name for name in transforms if name.startswith(("left", "right")) and any(part in name for part in _HAND_PARTS)]
        joints = np.stack([np.asarray(transforms[name][start : start + count, :3, 3], dtype=np.float64) for name in names], axis=1)
    source_h, source_w = frame_size(episode.with_suffix(".mp4"))
    height, width = int(source_h * SCALE), int(source_w * SCALE)
    rows = []
    for frame, camera in enumerate(c2w):
        world_to_camera = _invert(camera)
        points = joints[frame] @ world_to_camera[:3, :3].T + world_to_camera[:3, 3]
        depth = points[:, 2]
        front = depth > 0.05
        col = (intrinsic[0, 0] * points[front, 0] / depth[front] + intrinsic[0, 2]) * SCALE
        row = (intrinsic[1, 1] * points[front, 1] / depth[front] + intrinsic[1, 2]) * SCALE
        inside = (col >= 0) & (col < width) & (row >= 0) & (row < height)
        rows.append(np.stack([np.full(inside.sum(), frame), row[inside].astype(int), col[inside].astype(int), depth[front][inside]], axis=1))
    return np.concatenate(rows) if rows else np.zeros((0, 4))


def _hand_scale(depth: np.ndarray, hand: np.ndarray) -> float | None:
    """Scale that puts the visible hand at its tracked depth, or None with too few joints in view."""
    if len(hand) < _HAND_MIN:
        return None
    frame, row, col = (hand[:, index].astype(int) for index in range(3))
    ratio = hand[:, 3] / np.maximum(depth[frame, row, col], 1e-6)
    middle = np.median(ratio)
    # Joints hidden behind an object read the occluder, so keep the ones near the consensus.
    near = ratio[np.abs(ratio / middle - 1) < 0.2]
    # A split consensus can leave none near the median; NaN would spread to the whole piece.
    if len(near) == 0 or not np.isfinite(middle):
        return None
    return float(np.median(near))


def prepare_view(episode: Path, frames: np.ndarray, start: int = 0) -> dict:
    """CPU side of DA3: resize and cameras for frames that begin at `start`. The GPU actor only forwards."""
    thirdparty.use_da3()
    from depth_anything_3.utils.io.input_processor import InputProcessor

    extrinsics, intrinsics = _cameras(Path(episode), frames.shape[1], frames.shape[2], start, len(frames))
    frames = frames[: len(extrinsics)]
    images, extrinsics_t, intrinsics_t = InputProcessor()(
        [frame for frame in frames], extrinsics, intrinsics, process_res=PROCESS_RES, num_workers=1
    )
    return {
        "images": images.cpu(),
        "extrinsics": extrinsics_t.cpu(),
        "intrinsics": intrinsics_t.cpu(),
        "size": _out_size(frames),
        "hand": _hand_points(Path(episode), start, len(extrinsics)),
    }


def _forward_ready(items, model) -> list[tuple[np.ndarray, float | None, float]]:
    """One DA3 forward per group of windows with the same view count and size.

    Each window comes back as (depth at the stored size, hand scale or None, camera-path scale).
    """
    thirdparty.use_da3()
    import torch
    from depth_anything_3.utils.geometry import affine_inverse
    from depth_anything_3.utils.io.output_processor import OutputProcessor

    made: list[tuple | None] = [None] * len(items)
    place = {id(item): index for index, item in enumerate(items)}
    buckets = {}
    for item in items:
        buckets.setdefault(tuple(item["key"]), []).append(item)
    for bucket in buckets.values():
        images = torch.stack([item["images"] for item in bucket]).cuda().float()
        extrinsics = torch.cat([_normalize_extrinsics(affine_inverse, item["extrinsics"]) for item in bucket], dim=0)
        intrinsics = torch.cat([item["intrinsics"][None] for item in bucket], dim=0).cuda().float()
        with torch.inference_mode():
            raw = model(images, extrinsics, intrinsics, ref_view_strategy="middle")
        batch = images.shape[0]
        for index, item in enumerate(bucket):
            prediction = OutputProcessor()(_slice_output(raw, index, batch))
            depth = np.asarray(prediction.depth, dtype=np.float32)
            height, width = item["size"]
            if depth.ndim == 4:
                depth = depth.squeeze(1)
            if depth.shape[1:] != (height, width):
                depth = np.stack([cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in depth])
            camera = 1.0 / _metric_scale(prediction.extrinsics, item["extrinsics"].numpy())
            made[place[id(item)]] = (depth, _hand_scale(depth, item["hand"]), camera)
    return made


def _slice_output(raw, index: int, batch: int):
    import torch

    data = {}
    for key, value in raw.items():
        if torch.is_tensor(value) and value.ndim >= 1 and value.shape[0] == batch:
            data[key] = value[index : index + 1]
        else:
            data[key] = value
    return data


def _metric_scale(predicted: np.ndarray, reference: np.ndarray) -> float:
    """Scale that takes predicted depth into the EgoDex camera trajectory's meters."""
    source = _centers(reference)
    target = _centers(predicted)
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)
    covariance = target_c.T @ source_c / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[-1, -1] = -1
    variance = float((source_c ** 2).sum() / len(source))
    if variance < 1e-8:
        return 1.0
    return float(np.sum(singular * np.diag(sign)) / variance)


def _centers(extrinsics: np.ndarray) -> np.ndarray:
    if extrinsics.shape[-2:] == (3, 4):
        padded = np.repeat(np.eye(4)[None], len(extrinsics), axis=0)
        padded[:, :3, :4] = extrinsics
        extrinsics = padded
    rotation = extrinsics[:, :3, :3]
    return -np.einsum("nij,nj->ni", rotation.transpose(0, 2, 1), extrinsics[:, :3, 3])


def _normalize_extrinsics(affine_inverse, extrinsics):
    import torch

    poses = extrinsics[None].cuda().float()
    transform = affine_inverse(poses[:, :1])
    normalized = poses @ transform
    cameras = affine_inverse(normalized)
    median = torch.clamp(cameras[..., :3, 3].norm(dim=-1).median(), min=1e-1)
    normalized = normalized.clone()
    normalized[..., :3, 3] = normalized[..., :3, 3] / median
    return normalized


def _invert(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse
