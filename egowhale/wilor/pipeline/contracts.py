"""Versioned, content-checked hand artifacts independent of installed models."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Artifact:
    arrays: dict[str, np.ndarray]
    meta: dict


def canonical(value):
    if isinstance(value, dict):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def artifact_signature(meta: dict, arrays: dict[str, np.ndarray]) -> str:
    clean = {k: v for k, v in meta.items() if k != 'signature'}
    digest = hashlib.sha256(json.dumps(canonical(clean), sort_keys=True, allow_nan=False).encode())
    for key, value in sorted(arrays.items()):
        value = np.asarray(value)
        if value.dtype.hasobject:
            raise ValueError(f'object array forbidden: {key}')
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)]).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def validate_artifact(artifact: Artifact) -> None:
    a, m = artifact.arrays, artifact.meta
    if not m.get('kind'):
        raise ValueError('artifact kind missing')
    if m.get('kind') == 'masks':
        files, hashes = m.get('files', []), m.get('sha256', [])
        if len(files) != len(hashes) or len(files) != len(a.get('frame_index', [])):
            raise ValueError('incomplete mask references')
        for file, expected in zip(files, hashes):
            try:
                actual = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            except OSError as error:
                raise ValueError(f'missing external mask: {file}') from error
            if actual != expected:
                raise ValueError(f'external mask content changed: {file}')
    if 'intrinsic' in a:
        k = a['intrinsic']
        if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
                or not np.allclose(k[2], [0, 0, 1]) or abs(k[0, 1]) > 1e-8 or abs(k[1, 0]) > 1e-8):
            raise ValueError('invalid intrinsic')
    if 'frame_index' in a:
        index = a['frame_index']
        if index.ndim != 1 or index.dtype.kind not in 'iu' or (np.diff(index) <= 0).any() or (index < 0).any():
            raise ValueError('invalid frame_index')
    if 'w2c' in a:
        poses = a['w2c']
        if poses.shape != (len(a['frame_index']), 4, 4) or not np.isfinite(poses).all():
            raise ValueError('invalid w2c pose timeline')
        r = poses[:, :3, :3]
        if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-4)
                or not np.allclose(np.linalg.det(r), 1, atol=1e-4)):
            raise ValueError('invalid camera pose rotation')
    if 'fps' in a and (np.shape(a['fps']) != () or not np.isfinite(a['fps']) or a['fps'] <= 0):
        raise ValueError('invalid fps')
    if 'valid' in a:
        valid = a['valid']
        if valid.ndim != 2 or valid.shape[1] != 2 or valid.dtype != np.bool_:
            raise ValueError('invalid valid shape/dtype')
        t = len(valid)
        for key, shape in [('joints_camera', (t, 2, 21, 3)), ('keypoints_2d', (t, 2, 21, 2))]:
            if key not in a or a[key].shape != shape:
                raise ValueError(f'invalid {key} shape')
        points = a['joints_camera'][valid]
        if not np.isfinite(points).all() or (points[..., 2] <= 0).any():
            raise ValueError('invalid visible joints')
        if 'frame_index' not in a or not np.array_equal(a['frame_index'], np.arange(t)):
            raise ValueError('track frame_index must preserve complete timeline')
        for key in ('observed', 'interpolated', 'boundary_trimmed'):
            if key in a and (a[key].shape != valid.shape or a[key].dtype != np.bool_):
                raise ValueError(f'invalid {key}')
        if 'boundary_trimmed' in a and (a['boundary_trimmed'] & valid).any():
            raise ValueError('boundary-trimmed samples cannot be valid')
    if 'offsets' in a:
        offsets = a['offsets']
        if offsets.ndim != 1 or not len(offsets) or offsets[0] != 0 or (np.diff(offsets) < 0).any():
            raise ValueError('invalid candidate offsets')
        if int(offsets[-1]) != len(a['joints_camera']):
            raise ValueError('candidate count mismatch')


def write_artifact(path: Path, artifact: Artifact) -> None:
    path = Path(path)
    validate_artifact(artifact)
    meta = canonical({**artifact.meta, 'schema': 3})
    signature = artifact_signature(meta, artifact.arrays)
    meta['signature'] = signature
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = []
    try:
        for suffix in ('.npz', '.json'):
            fd, name = tempfile.mkstemp(dir=path.parent, prefix='.' + path.stem, suffix=suffix)
            os.close(fd)
            temp.append(Path(name))
        with temp[0].open('wb') as handle:
            np.savez_compressed(handle, **artifact.arrays, signature=np.array(signature))
        temp[1].write_text(json.dumps(meta, indent=2, allow_nan=False))
        os.replace(temp[0], path)
        os.replace(temp[1], path.with_suffix('.json'))
    finally:
        for p in temp:
            p.unlink(missing_ok=True)
    artifact.meta = meta


def read_artifact(path: Path, *, expected_kind: str) -> Artifact:
    path = Path(path)
    if not path.is_file() or not path.with_suffix('.json').is_file():
        raise ValueError(f'incomplete artifact: {path}')
    meta = json.loads(path.with_suffix('.json').read_text())
    with np.load(path, allow_pickle=False) as data:
        signature = str(data['signature'])
        arrays = {k: np.asarray(data[k]) for k in data.files if k != 'signature'}
    if meta.get('schema') != 3 or meta.get('kind') != expected_kind:
        raise ValueError('artifact schema/kind mismatch')
    if signature != meta.get('signature') or signature != artifact_signature(meta, arrays):
        raise ValueError('artifact content signature mismatch')
    artifact = Artifact(arrays, meta)
    validate_artifact(artifact)
    return artifact


def derive(source: Artifact, kind: str, **meta) -> Artifact:
    return Artifact({k: v.copy() for k, v in source.arrays.items()},
                    {**{k: v for k, v in source.meta.items() if k != 'signature'}, 'kind': kind,
                     'upstream_signature': source.meta.get('signature'), **meta})
