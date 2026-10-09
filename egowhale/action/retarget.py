"""Hand keypoints to a smoothed camera-frame parallel gripper."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.signal import savgol_coeffs, savgol_filter
from scipy.spatial.transform import Rotation

from egowhale.step import GRIPPER, Step

FPS = 30.0
WINDOW = 21
POLY = 3
SIGMA = 10.0
_TIPS = ("Hand", "ThumbTip", "IndexFingerTip", "MiddleFingerTip")


class Retarget(Step):
    name = "retarget"
    makes = (GRIPPER,)
    gpus = 1

    def run(self, src: Path, dst: Path) -> None:
        save_gripper(Path(dst), self.compute(read_episode(Path(src))))

    def compute(self, raw: dict) -> dict:
        position, rotation, width, valid, intrinsic = _episode_gpu(raw)
        position, rotation, width = _smooth_gpu(position, rotation, width, valid)
        return {
            "position": position.astype(np.float32),
            "rotation": rotation.astype(np.float32),
            "width": width.astype(np.float32),
            "valid": valid,
            "intrinsic": intrinsic.astype(np.float32),
        }


def _episode(path: Path):
    with h5py.File(path, "r") as handle:
        intrinsic = np.asarray(handle["camera/intrinsic"], dtype=np.float64)
        transforms = handle["transforms"]
        camera = np.asarray(transforms["camera"], dtype=np.float64)
        joints = {
            name: np.asarray(dataset, dtype=np.float64)
            for name, dataset in transforms.items()
            if name != "camera"
        }
    frames = camera.shape[0]
    position = np.full((frames, 2, 3), np.nan, dtype=np.float64)
    rotation = np.full((frames, 2, 3, 3), np.nan, dtype=np.float64)
    width = np.full((frames, 2), np.nan, dtype=np.float64)
    valid = np.zeros((frames, 2), dtype=bool)
    for frame in range(frames):
        cam_from_world = _invert(camera[frame])
        points = {name: value[frame, :3, 3] for name, value in joints.items()}
        for side, name in enumerate(("left", "right")):
            pose = _hand(points, name)
            if pose is None:
                continue
            world_p, world_r, opening = pose
            position[frame, side] = cam_from_world[:3, :3] @ world_p + cam_from_world[:3, 3]
            rotation[frame, side] = cam_from_world[:3, :3] @ world_r
            width[frame, side] = opening
            valid[frame, side] = True
    return position, rotation, width, valid, intrinsic


def _hand(points: dict[str, np.ndarray], side: str):
    names = [f"{side}{tip}" for tip in _TIPS]
    if any(name not in points for name in names):
        return None
    wrist, thumb, index, middle = (points[name] for name in names)
    virtual = 0.7 * index + 0.3 * middle
    jaw = thumb - virtual
    opening = float(np.linalg.norm(jaw))
    if opening < 1e-4:
        return None
    sign = 1.0 if side == "right" else -1.0
    grasp = sign * jaw / opening
    normal = np.cross(grasp, virtual - wrist)
    norm = np.linalg.norm(normal)
    if norm < 1e-4:
        return None
    normal /= norm
    approach = np.cross(normal, grasp)
    return 0.5 * (thumb + virtual), np.column_stack((approach, normal, grasp)), opening


def _invert(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _smooth(position, rotation, width, valid):
    position, rotation, width = position.copy(), rotation.copy(), width.copy()
    for side in range(2):
        for start, end in _segments(valid[:, side]):
            window = _window(end - start)
            if window is not None:
                position[start:end, side] = savgol_filter(position[start:end, side], window, POLY, axis=0)
                width[start:end, side] = np.maximum(0.0, savgol_filter(width[start:end, side], window, POLY))
            rotation[start:end, side] = _slerp_smooth(rotation[start:end, side])
    return position, rotation, width


def _segments(valid: np.ndarray) -> list[tuple[int, int]]:
    spans, start = [], None
    for index, flag in enumerate(valid):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append((start, index))
            start = None
    if start is not None:
        spans.append((start, len(valid)))
    return spans


def _window(length: int) -> int | None:
    window = min(WINDOW, length if length % 2 else length - 1)
    return window if window > POLY else None


def _slerp_smooth(rotations: np.ndarray) -> np.ndarray:
    quats = Rotation.from_matrix(rotations).as_quat()
    radius = WINDOW // 2
    out = np.empty_like(rotations)
    for center in range(len(rotations)):
        indices = range(max(0, center - radius), min(len(rotations), center + radius + 1))
        mean = quats[center].copy()
        weight_sum = 0.0
        for index in sorted(indices, key=lambda item: abs(item - center)):
            sample = quats[index].copy()
            if np.dot(mean, sample) < 0.0:
                sample *= -1.0
            weight = float(np.exp(-0.5 * ((index - center) / SIGMA) ** 2))
            if weight_sum == 0.0:
                mean, weight_sum = sample, weight
                continue
            mean = _slerp(mean, sample, weight / (weight_sum + weight))
            weight_sum += weight
        out[center] = Rotation.from_quat(mean).as_matrix()
    return out


def read_episode(path: Path) -> dict:
    with h5py.File(path, "r") as handle:
        transforms = handle["transforms"]
        return {
            "intrinsic": np.asarray(handle["camera/intrinsic"], dtype=np.float64),
            "camera": np.asarray(transforms["camera"], dtype=np.float64),
            "joints": {
                name: np.asarray(dataset, dtype=np.float64)
                for name, dataset in transforms.items()
                if name != "camera"
            },
        }


def save_gripper(dst: Path, result: dict) -> None:
    path = Path(dst) / GRIPPER
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        position=result["position"],
        rotation=result["rotation"],
        width=result["width"],
        valid=result["valid"],
        intrinsic=result["intrinsic"],
        fps=np.float32(FPS),
    )


def _episode_gpu(raw: dict):
    """Same geometry as `_episode`, stacked over frames on GPU."""
    intrinsic = raw["intrinsic"]
    camera = torch.tensor(raw["camera"], device="cuda")
    joints = {name: torch.tensor(value, device="cuda") for name, value in raw["joints"].items()}
    frames = camera.shape[0]
    rotation_c = camera[:, :3, :3].transpose(-1, -2)
    translation = -torch.einsum("tij,tj->ti", rotation_c, camera[:, :3, 3])
    position = torch.full((frames, 2, 3), torch.nan, device="cuda")
    rotation = torch.full((frames, 2, 3, 3), torch.nan, device="cuda")
    width = torch.full((frames, 2), torch.nan, device="cuda")
    valid = torch.zeros((frames, 2), dtype=torch.bool, device="cuda")
    for side, name in enumerate(("left", "right")):
        names = [f"{name}{tip}" for tip in _TIPS]
        if any(tip not in joints for tip in names):
            continue
        wrist, thumb, index, middle = (joints[tip][:, :3, 3] for tip in names)
        virtual = 0.7 * index + 0.3 * middle
        jaw = thumb - virtual
        opening = jaw.norm(dim=-1)
        sign = 1.0 if name == "right" else -1.0
        grasp = sign * jaw / opening.clamp_min(1e-12).unsqueeze(-1)
        normal = torch.cross(grasp, virtual - wrist, dim=-1)
        scale = normal.norm(dim=-1)
        good = (opening >= 1e-4) & (scale >= 1e-4)
        normal = normal / scale.clamp_min(1e-12).unsqueeze(-1)
        approach = torch.cross(normal, grasp, dim=-1)
        world_p = 0.5 * (thumb + virtual)
        world_r = torch.stack((approach, normal, grasp), dim=-1)
        position[:, side] = torch.einsum("tij,tj->ti", rotation_c, world_p) + translation
        rotation[:, side] = torch.einsum("tij,tjk->tik", rotation_c, world_r)
        width[:, side] = opening
        position[~good, side] = torch.nan
        rotation[~good, side] = torch.nan
        width[~good, side] = torch.nan
        valid[:, side] = good
    return (
        position.cpu().numpy(),
        rotation.cpu().numpy(),
        width.cpu().numpy(),
        valid.cpu().numpy(),
        intrinsic,
    )


def _smooth_gpu(position, rotation, width, valid):
    """Same smoother as `_smooth`, with the filters evaluated on GPU."""
    position = torch.tensor(position, device="cuda")
    rotation = torch.tensor(rotation, device="cuda")
    width = torch.tensor(width, device="cuda")
    for side in range(2):
        for start, end in _segments(valid[:, side]):
            window = _window(end - start)
            if window is not None:
                position[start:end, side] = _savgol(position[start:end, side], window)
                width[start:end, side] = _savgol(width[start:end, side].unsqueeze(-1), window).squeeze(-1).clamp_min(0.0)
            rotation[start:end, side] = _slerp_smooth_gpu(rotation[start:end, side])
    return position.cpu().numpy(), rotation.cpu().numpy(), width.cpu().numpy()


def _savgol(values: torch.Tensor, window: int) -> torch.Tensor:
    coeffs = torch.tensor(savgol_coeffs(window, POLY), dtype=values.dtype, device=values.device)
    flat = values.reshape(values.shape[0], -1).transpose(0, 1).unsqueeze(1)
    kernel = coeffs.flip(0).view(1, 1, -1)
    with torch.backends.cudnn.flags(enabled=False):  # cuDNN plans each new length (~0.3 s)
        filtered = torch.nn.functional.conv1d(flat, kernel, padding=window // 2)
    filtered = filtered.squeeze(1).transpose(0, 1).reshape(values.shape)
    return _savgol_edges(values, filtered, window)


def _savgol_edges(values: torch.Tensor, filtered: torch.Tensor, window: int) -> torch.Tensor:
    half = window // 2
    out = filtered.clone()
    out[:half] = _poly_edge(values[:window], torch.arange(half, device=values.device))
    out[-half:] = _poly_edge(values[-window:], torch.arange(window - half, window, device=values.device))
    return out


def _poly_edge(samples: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    degree = torch.arange(POLY, -1, -1, device=samples.device, dtype=samples.dtype)
    basis = torch.arange(samples.shape[0], device=samples.device, dtype=samples.dtype)
    design = basis.unsqueeze(1) ** degree
    target = samples.reshape(samples.shape[0], -1)
    coeff = torch.linalg.lstsq(design, target).solution
    query = index.to(samples.dtype).unsqueeze(1) ** degree
    return (query @ coeff).reshape((-1, *samples.shape[1:]))


def _slerp_smooth_gpu(rotations: torch.Tensor) -> torch.Tensor:
    """One SLERP step is shared by every frame. Only the window radius stays sequential."""
    quats = _quat_from_matrix(rotations)
    count = quats.shape[0]
    mean = quats
    weight_sum = torch.ones(count, dtype=quats.dtype, device=quats.device)
    centers = torch.arange(count, device=quats.device)
    for offset in range(1, WINDOW // 2 + 1):
        weight = torch.exp(torch.tensor(-0.5 * (offset / SIGMA) ** 2, dtype=quats.dtype, device=quats.device))
        for shift in (-offset, offset):
            index = centers + shift
            inside = (index >= 0) & (index < count)
            sample = quats[index.clamp(0, count - 1)]
            mean, weight_sum = _slerp_into(mean, weight_sum, sample, weight, inside)
    return _matrix_from_quat(mean)


def _slerp_into(mean, weight_sum, sample, weight, inside):
    sign = torch.where((mean * sample).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    sample = torch.where(inside.unsqueeze(-1), sign * sample, mean)
    fraction = torch.where(inside, weight / (weight_sum + weight), 0.0)
    blended = _slerp_gpu(mean, sample, fraction)
    return blended, weight_sum + torch.where(inside, weight, 0.0)


def _quat_from_matrix(rotations: torch.Tensor) -> torch.Tensor:
    trace = rotations[..., 0, 0] + rotations[..., 1, 1] + rotations[..., 2, 2]
    candidates = (
        _quat_branch(trace + 1.0, rotations[..., 2, 1] - rotations[..., 1, 2], rotations[..., 0, 2] - rotations[..., 2, 0], rotations[..., 1, 0] - rotations[..., 0, 1], 0),
        _quat_branch(1.0 + rotations[..., 0, 0] - rotations[..., 1, 1] - rotations[..., 2, 2], rotations[..., 2, 1] - rotations[..., 1, 2], rotations[..., 0, 1] + rotations[..., 1, 0], rotations[..., 0, 2] + rotations[..., 2, 0], 1),
        _quat_branch(1.0 + rotations[..., 1, 1] - rotations[..., 0, 0] - rotations[..., 2, 2], rotations[..., 0, 2] - rotations[..., 2, 0], rotations[..., 0, 1] + rotations[..., 1, 0], rotations[..., 1, 2] + rotations[..., 2, 1], 2),
        _quat_branch(1.0 + rotations[..., 2, 2] - rotations[..., 0, 0] - rotations[..., 1, 1], rotations[..., 1, 0] - rotations[..., 0, 1], rotations[..., 0, 2] + rotations[..., 2, 0], rotations[..., 1, 2] + rotations[..., 2, 1], 3),
    )
    primary = (rotations[..., 0, 0] > rotations[..., 1, 1]) & (rotations[..., 0, 0] > rotations[..., 2, 2])
    secondary = rotations[..., 1, 1] > rotations[..., 2, 2]
    quat = torch.where(
        trace.unsqueeze(-1) > 0,
        candidates[0],
        torch.where(primary.unsqueeze(-1), candidates[1], torch.where(secondary.unsqueeze(-1), candidates[2], candidates[3])),
    )
    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _quat_branch(scale, w, x, y, slot: int) -> torch.Tensor:
    scale = torch.sqrt(scale.clamp_min(0.0)) * 2.0
    safe = scale.clamp_min(1e-12)
    values = (0.25 * scale, w / safe, x / safe, y / safe)
    order = ((1, 2, 3, 0), (0, 2, 3, 1), (2, 0, 3, 1), (2, 3, 0, 1))[slot]
    return torch.stack(tuple(values[index] for index in order), dim=-1)


def _matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    x, y, z, w = quat.unbind(-1)
    return torch.stack((
        torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
        torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1),
        torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1),
    ), dim=-2)


def _slerp_gpu(first: torch.Tensor, second: torch.Tensor, fraction: torch.Tensor) -> torch.Tensor:
    dot = (first * second).sum(-1).clamp(-1.0, 1.0)
    near = dot > 0.9995
    angle = torch.acos(dot)
    sine = torch.sin(angle).clamp_min(1e-12)
    span = fraction.unsqueeze(-1)
    blended = (torch.sin((1.0 - fraction) * angle).unsqueeze(-1) * first + torch.sin(fraction * angle).unsqueeze(-1) * second) / sine.unsqueeze(-1)
    mixed = first + span * (second - first)
    mixed = mixed / mixed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return torch.where(near.unsqueeze(-1), mixed, blended)


def _slerp(first: np.ndarray, second: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.clip(np.dot(first, second), -1.0, 1.0))
    if dot < 0.0:
        second, dot = -second, -dot
    if dot > 0.9995:
        mixed = first + fraction * (second - first)
        return mixed / np.linalg.norm(mixed)
    angle = np.arccos(dot)
    return (
        np.sin((1.0 - fraction) * angle) * first + np.sin(fraction * angle) * second
    ) / np.sin(angle)
