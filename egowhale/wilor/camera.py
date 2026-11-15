"""Explicit camera assumptions for monocular hand reconstruction (OpenCV axes)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def resolve_camera(width: int, height: int, mode: str, *, focal_ratio: float,
                   calibration: Path | None = None) -> tuple[np.ndarray, dict]:
    if min(width, height) <= 0:
        raise ValueError('invalid image size')
    meta = {'source': mode, 'width': width, 'height': height,
            'axes': 'x right, y down, z forward', 'extrinsics': 'unknown; per-frame camera coordinates'}
    if mode == 'wilor-default':
        f = focal_ratio * max(width, height)
        k = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1]], dtype=float)
        meta['assumed'] = True
    elif mode in ('provided', 'egodex'):
        if calibration is None:
            raise ValueError(f'{mode} requires an explicit calibration path')
        calibration = Path(calibration).resolve()
        if mode == 'egodex':
            from egowhale.media import video_shape
            source_video = calibration.with_suffix('.mp4')
            if not source_video.is_file():
                raise ValueError('EgoDex calibration needs its sibling video to verify image size; otherwise provide a sized camera JSON')
            _, source_h, source_w, _ = video_shape(source_video)
            if (source_w, source_h) != (width, height):
                raise ValueError('EgoDex calibration image size differs from input video; explicitly rescale K in a camera JSON')
            import h5py
            with h5py.File(calibration) as h:
                k = np.asarray(h['camera/intrinsic'], dtype=float)
        else:
            obj = json.loads(calibration.read_text())
            if (obj['width'], obj['height']) != (width, height):
                raise ValueError('calibration image size differs from video; explicitly rescale K first')
            k = np.asarray(obj['intrinsic'], dtype=float)
        meta.update(calibration=str(calibration), assumed=False)
    else:
        raise ValueError(f'unknown camera mode {mode}')
    if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not np.allclose(k[2], [0, 0, 1]) or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('K must be finite, zero-skew, with positive fx/fy and last row [0,0,1]')
    meta['depth_focal_rule'] = 'sqrt(fx*fy); monocular approximation'
    meta['intrinsic'] = k.tolist()
    return k, meta


def crop_camera_to_full(camera: np.ndarray, center: np.ndarray, size: np.ndarray,
                        intrinsic: np.ndarray) -> np.ndarray:
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    depth = 2 * np.sqrt(fx * fy) / (size * camera[:, 0] + 1e-9)
    x = (center[:, 0] - intrinsic[0, 2]) * depth / fx + camera[:, 1]
    y = (center[:, 1] - intrinsic[1, 2]) * depth / fy + camera[:, 2]
    return np.stack((x, y, depth), axis=-1)


def project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    pixels = points @ intrinsic.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return pixels[..., :2] / pixels[..., 2:]
