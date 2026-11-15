"""Explicit camera calibration and monocular, arbitrary-scale camera motion."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from egowhale.media import video_shape
from egowhale.wilor.camera import resolve_camera
from egowhale.wilor.step import WorkerClient

from .contracts import Artifact, read_artifact


def restore_intrinsic(k_processed: np.ndarray, image_from_original: np.ndarray) -> np.ndarray:
    return np.linalg.solve(image_from_original, k_processed)


def summarize_intrinsics(samples, width, height, tolerance):
    k = np.median(samples, axis=0)
    focal = samples[:, [0, 1], [0, 1]]
    spread = (np.percentile(focal, 95, axis=0) - np.percentile(focal, 5, axis=0)) / np.median(focal, axis=0)
    if not np.isfinite(samples).all() or (focal <= 0).any() or (spread > tolerance).any():
        raise ValueError(f'DA3 intrinsic unstable: relative focal spread {spread}')
    if not (0 <= k[0, 2] <= width and 0 <= k[1, 2] <= height):
        raise ValueError('DA3 principal point outside original image')
    return k


def stitch_camera_windows(windows: list[Artifact], frame_count: int) -> Artifact:
    c2w = np.full((frame_count, 4, 4), np.nan)
    for window in windows:
        ids = window.arrays['frame_index']
        local = np.linalg.inv(window.arrays['w2c'])
        overlap = np.isfinite(c2w[ids, 0, 0])
        if overlap.any():
            reference = c2w[ids[overlap]]
            moving = local[overlap]
            align = Rotation.from_matrix(reference[:, :3, :3] @ moving[:, :3, :3].transpose(0, 2, 1)).mean().as_matrix()
            x = moving[:, :3, 3] @ align.T
            y = reference[:, :3, 3]
            xc, yc = x - x.mean(0), y - y.mean(0)
            denominator = (xc * xc).sum()
            if denominator < 1e-10 or (yc * yc).sum() < 1e-10:
                raise ValueError('insufficient parallax for camera window scale alignment')
            scale = (xc * yc).sum() / denominator
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError('invalid camera window scale')
            translation = y.mean(0) - scale * x.mean(0)
            local[:, :3, :3] = align @ local[:, :3, :3]
            local[:, :3, 3] = scale * (local[:, :3, 3] @ align.T) + translation
        c2w[ids[~overlap]] = local[~overlap]
    if not np.isfinite(c2w).all():
        raise ValueError('camera windows do not cover full timeline')
    c2w[:, :3, :3] = Rotation.from_matrix(c2w[:, :3, :3]).as_matrix()
    return Artifact({'w2c': np.linalg.inv(c2w), 'frame_index': np.arange(frame_count),
                     'intrinsic': windows[0].arrays['intrinsic']},
                    {'kind': 'camera', 'pose_scale': 'arbitrary', 'reference_frame': 0,
                     'axes': 'OpenCV x-right y-down z-forward', 'pose_convention': 'world_to_camera'})


def estimate_camera(video: Path, options: dict) -> Artifact:
    count, height, width, fps = video_shape(video)
    source = options['source']
    known = None
    known_meta = {}
    if source != 'da3':
        known, meta = resolve_camera(width, height, source, focal_ratio=5000 / 256,
                                     calibration=Path(options['calibration']) if options.get('calibration') else None)
        known_meta = meta
        if not options.get('motion'):
            return Artifact({'intrinsic': known, 'frame_index': np.arange(count), 'fps': np.array(fps)},
                            {'kind': 'camera', **meta, 'pose_scale': 'unavailable'})
    client = WorkerClient([str(Path(options['python']).absolute()), str(Path(__file__).parent / 'workers/da3.py')],
                          Path(options['repo']), options['timeout'])
    try:
        response = client.request({'video': str(Path(video).resolve()), 'out': str(Path(options['out']).resolve()),
                                   'options': {**options, 'known_intrinsic': None if known is None else known.tolist()}})
        result = read_artifact(Path(response['artifact']), expected_kind='camera')
    finally:
        client.close()
    result.meta.update(source=source, width=width, height=height, assumed=known_meta.get('assumed', False))
    if known_meta:
        result.meta['intrinsic_source'] = known_meta
    return result
