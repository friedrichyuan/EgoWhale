"""Read raw EgoDex hand annotations for the hand-video Path A."""
from pathlib import Path

import h5py
import numpy as np

# Same 21-joint order as hand_processor's MANO skeleton.
JOINTS = ['Hand', *(f'{finger}Finger{s}' for finger in ('Index', 'Middle', 'Ring', 'Little')
                    for s in ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip')),
          *(f'Thumb{s}' for s in ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip'))]


def inspect_episode(path):
    if not Path(path).is_file():
        return False, False
    with h5py.File(path, 'r') as handle:
        has_intrinsics = 'camera/intrinsic' in handle
        has_labels = ('transforms/camera' in handle and any(
            all(f'transforms/{side}{joint}' in handle for joint in JOINTS)
            for side in ('left', 'right')))
    return has_intrinsics, has_labels


def read_hands(path, count, intrinsic):
    """Return camera-space keypoints and confidence, preserving input frame order."""
    result = {}
    with h5py.File(path, 'r') as handle:
        cameras = np.asarray(handle['transforms/camera'][:count], dtype=float)
        if cameras.shape != (count, 4, 4):
            raise ValueError('EgoDex 相机轨迹帧数与视频不一致')
        for side in ('left', 'right'):
            names = [f'{side}{joint}' for joint in JOINTS]
            if not all(f'transforms/{name}' in handle for name in names):
                result[side] = (np.zeros((count, 21, 3)), np.zeros((count, 21, 2)), np.zeros(count, bool))
                continue
            world = np.stack([np.asarray(handle[f'transforms/{name}'][:count, :3, 3], dtype=float)
                              for name in names], axis=1)
            if world.shape != (count, 21, 3):
                raise ValueError(f'EgoDex {side} 手部标注帧数与视频不一致')
            xyz = np.einsum('tji,tkj->tki', cameras[:, :3, :3], world - cameras[:, None, :3, 3])
            confidence = np.stack([np.asarray(handle[f'confidences/{name}'][:count])
                                   if f'confidences/{name}' in handle else np.ones(count)
                                   for name in names], axis=1)
            valid = np.isfinite(xyz).all(axis=-1) & (xyz[..., 2] > 0) & (confidence >= .5)
            uv = np.zeros((count, 21, 2), dtype=float)
            projected = np.einsum('ij,tkj->tki', intrinsic, xyz)
            np.divide(projected[..., :2], projected[..., 2:3], out=uv, where=valid[..., None])
            valid &= np.isfinite(uv).all(axis=-1)
            uv[~valid] = 0
            result[side] = (xyz, uv, valid.any(axis=1))
    return result
