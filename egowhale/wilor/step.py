"""Optional hand reconstruction through an isolated, persistent WiLoR process."""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import signal
import subprocess
import time
from pathlib import Path

import numpy as np

from egowhale.media import video_path
from egowhale.paths import HANDS_WILOR, HANDS_WILOR_META, ROOT, THIRDPARTY
from egowhale.step import Step
from egowhale.wilor.retarget_export import RETARGET_HDF5, RETARGET_VIDEO, validate_retarget
from egowhale.wilor.camera import project


def file_identity(path: Path) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def pack_detections(frames: list, intrinsic: np.ndarray, fps: float) -> dict:
    count = len(frames)
    joints = np.full((count, 2, 21, 3), np.nan, np.float32)
    confidence = np.zeros((count, 2), np.float32)
    bbox = np.full((count, 2, 4), np.nan, np.float32)
    valid = np.zeros((count, 2), bool)
    for frame, detections in enumerate(frames):
        for row in detections:
            side = int(row['right'])
            points = np.asarray(row['joints'])
            if row['confidence'] <= confidence[frame, side] or not np.isfinite(points).all() or np.any(points[:, 2] <= 0):
                continue
            joints[frame, side] = points
            confidence[frame, side] = row['confidence']
            bbox[frame, side] = row['bbox']
            valid[frame, side] = True
    return {'joints_camera': joints, 'keypoints_2d': project(joints, intrinsic).astype(np.float32),
            'confidence': confidence, 'bbox': bbox, 'valid': valid, 'intrinsic': intrinsic,
            'frame_index': np.arange(count), 'timestamps': np.arange(count) / fps, 'fps': np.float32(fps)}


def validate_legacy_prediction(result, count, *, require_positive_depth=True) -> None:
    if not isinstance(count, int) or count < 1:
        raise ValueError('WiLoR output invalid frame count')
    shapes = {'joints_camera': (count, 2, 21, 3), 'keypoints_2d': (count, 2, 21, 2),
              'valid': (count, 2), 'confidence': (count, 2), 'bbox': (count, 2, 4),
              'frame_index': (count,), 'timestamps': (count,), 'intrinsic': (3, 3), 'fps': ()}
    if any(key not in result for key in shapes):
        raise ValueError('WiLoR output missing required fields')
    if any(result[key].shape != shape for key, shape in shapes.items()):
        raise ValueError('WiLoR output shape mismatch')
    fps = float(result['fps'])
    if not np.isfinite(fps) or fps <= 0 or result['valid'].dtype != np.bool_:
        raise ValueError('WiLoR output invalid fps/valid fields')
    if not np.array_equal(result['frame_index'], np.arange(count)):
        raise ValueError('WiLoR output frame indices mismatch')
    if not np.allclose(result['timestamps'], np.arange(count) / fps):
        raise ValueError('WiLoR output timestamps mismatch')
    points = result['joints_camera'][result['valid']]
    if not np.isfinite(points).all() or (require_positive_depth and np.any(points[..., 2] <= 0)):
        raise ValueError('WiLoR output invalid visible joints')


def validate_prediction_input(src: Path, dst: Path, options: dict | None = None) -> dict:
    meta = json.loads((Path(dst) / HANDS_WILOR_META).read_text())
    if meta.get('pipeline'):
        from egowhale.wilor.pipeline.runner import validate_published
        return validate_published(src, dst, options)
    if meta.get('input') != file_identity(video_path(src)):
        raise ValueError('WiLoR input changed; use a new output directory')
    with np.load(Path(dst) / HANDS_WILOR) as result:
        if str(result['signature']) != meta['signature']:
            raise ValueError('WiLoR NPZ/metadata mismatch')
        validate_legacy_prediction(result, meta.get('frames'), require_positive_depth=meta.get('source') != 'ego2robot_sync')
    if meta.get('retarget_hdf5'):
        validate_retarget(Path(dst), meta['signature'], meta['frames'], hdf5=meta['retarget_hdf5'])
    marked = meta.get('calibration_input')
    if marked is not None and file_identity(Path(marked['path'])) != marked:
        raise ValueError('WiLoR calibration input changed; use a new output directory')
    marked_json = meta.get('intrinsics_json_input')
    if marked_json is not None and file_identity(Path(marked_json['path'])) != marked_json:
        raise ValueError('WiLoR JSON intrinsics changed; use a new output directory')
    for key, value in (options or {}).items():
        if key not in ('camera_mode', 'calibration', 'confidence'):
            continue
        if key == 'calibration' and value is not None:
            value = str(Path(value).resolve())
        if meta['options'].get(key) != value:
            raise ValueError('WiLoR camera/options changed; use a new output directory')
    return meta


def protect_worker_parent() -> None:
    """Linux worker guard: cancellation kills only this worker's own process group."""
    import ctypes
    import sys

    if sys.platform != 'linux' or os.getpgrp() != os.getpid():
        return
    expected = int(os.environ.get('EGOWHALE_WORKER_PARENT', os.getppid()))

    def terminate_group(_signum, _frame):
        os.killpg(os.getpgrp(), signal.SIGKILL)

    signal.signal(signal.SIGTERM, terminate_group)
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), 'cannot set model worker parent-death signal')
    if os.getppid() != expected:
        terminate_group(None, None)


class WorkerClient:
    def __init__(self, command: list[str], cwd: Path, timeout: float):
        self.command, self.cwd, self.timeout = command, cwd, timeout
        self.process = None
        self.busy = False

    def request(self, payload: dict) -> dict:
        if self.process is None:
            self.process = subprocess.Popen(self.command, cwd=self.cwd, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, text=True, bufsize=1, start_new_session=True,
                                            env={**os.environ, 'EGOWHALE_WORKER_PARENT': str(os.getpid())})
        self.busy = True
        try:
            self.process.stdin.write(json.dumps(payload) + '\n')
            self.process.stdin.flush()
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdout, selectors.EVENT_READ)
                deadline = time.monotonic() + self.timeout
                line = bytearray()
                while b'\n' not in line:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise TimeoutError(f'WiLoR worker exceeded {self.timeout}s')
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError(f'WiLoR worker exited ({self.process.poll()})')
                    line.extend(chunk)
                    if len(line) > 1024 * 1024:
                        raise RuntimeError('WiLoR response exceeds protocol limit')
            response = json.loads(line)
            if not response.get('ok'):
                raise RuntimeError(response.get('error', 'WiLoR worker failed'))
            self.busy = False
            return response
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is None:
            return
        process, self.process = self.process, None
        if not self.busy and process.poll() is None:
            process.stdin.close()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if not process.stdin.closed:
            process.stdin.close()
        process.stdout.close()
        self.busy = False


class WiLoR(Step):
    name = 'wilor'
    makes = (HANDS_WILOR, HANDS_WILOR_META, RETARGET_HDF5, RETARGET_VIDEO)
    gpus = 1

    def __init__(self, camera_mode='wilor-default', calibration=None, python=None, confidence=.3, timeout=1800, pipeline_config: str | Path | None = None, intrinsics_json=None):
        super().__init__()
        from egowhale.wilor.synced import CONFIG, DEFAULT_PYTHON
        self.options = {'camera_mode': camera_mode, 'calibration': str(Path(calibration).resolve()) if calibration else None,
                        'confidence': confidence}
        self.python = Path(python) if python else DEFAULT_PYTHON
        self.intrinsics_json = str(Path(intrinsics_json).resolve()) if intrinsics_json else None
        self.timeout = timeout
        self.client = None
        self.pipeline_config = Path(pipeline_config).absolute() if pipeline_config else CONFIG
        if not 0 < confidence < 1 or timeout <= 0:
            raise ValueError('confidence must be in (0,1) and timeout must be positive')

    def signature(self, src: Path) -> str:
        from egowhale.wilor.synced import signature
        return signature(src, self.pipeline_config, self.options['calibration'],
                         self.options['camera_mode'], self.options['confidence'], self.intrinsics_json)

    def validate_outputs(self, src, dst):
        path, meta = Path(dst) / HANDS_WILOR, Path(dst) / HANDS_WILOR_META
        if not path.exists() and not meta.exists():
            return
        if not path.is_file() or not meta.is_file():
            raise ValueError('incomplete WiLoR output; use a new output directory')
        marked = json.loads(meta.read_text())
        if marked.get('signature') != self.signature(src):
            raise ValueError('WiLoR configuration/input changed; use a new output directory')
        with np.load(path) as result:
            if str(result['signature']) != marked['signature']:
                raise ValueError('WiLoR NPZ/metadata mismatch')
            validate_legacy_prediction(result, marked.get('frames'), require_positive_depth=marked.get('source') != 'ego2robot_sync')
        if marked.get('retarget_hdf5'):
            validate_retarget(Path(dst), marked['signature'], marked['frames'], hdf5=marked['retarget_hdf5'])

    def load(self):
        if not self.python.is_file():
            raise FileNotFoundError(f'WiLoR Python not found: {self.python}')

    def run(self, src: Path, dst: Path):
        from egowhale.wilor.synced import require_mp4
        src = require_mp4(src)
        self.validate_outputs(src, dst)
        if (Path(dst) / HANDS_WILOR_META).is_file():
            return
        from egowhale.wilor.synced import run
        run(src, dst, self.pipeline_config, self.python, self.options['camera_mode'],
            self.options['calibration'], self.options['confidence'], self.intrinsics_json)
        self.validate_outputs(src, dst)

    def release(self):
        if self.client:
            self.client.close()
            self.client = None


def save_prediction(dst: Path, result: dict, metadata: dict):
    dst = Path(dst)
    path = dst / HANDS_WILOR
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {**result, 'signature': np.array(metadata['signature'])}
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as handle:
        np.savez_compressed(handle, **result)
    os.replace(temp, path)
    meta = dst / HANDS_WILOR_META
    temp = meta.with_suffix('.tmp')
    temp.write_text(json.dumps(metadata, indent=2, allow_nan=False))
    os.replace(temp, meta)
