"""EgoWhale adapter for the locally synchronized ego2robot hand processor."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np

SYNC_ROOT = Path(__file__).resolve().parent / 'ego2robot_sync'
CONFIG = Path(__file__).resolve().parent / 'cfgs/wilor.yaml'
RUNNER = SYNC_ROOT / 'scripts/run_hand_experiment.py'
DEFAULT_PYTHON = Path(__file__).resolve().parents[3] / '.conda-egohand/bin/python'


def identity(path):
    p = Path(path).resolve()
    s = p.stat()
    return {'path': str(p), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns}


def require_mp4(src):
    """Annotated HDF5 episodes belong to Retarget, not the prediction node."""
    src = Path(src).resolve()
    if src.suffix.lower() != '.mp4':
        raise ValueError('WiLoR 仅接受 MP4；有标注的 HDF5 请直接输入 Retarget')
    return src


def signature(src, config, calibration, camera_mode, confidence, intrinsics_json=None):
    src = require_mp4(src)
    files = [identity(src), identity(config)]
    if calibration and Path(calibration).resolve() != src:
        files.append(identity(calibration))
    if intrinsics_json:
        files.append(identity(intrinsics_json))
    thirdparty = Path(os.environ.get('EGOWHALE_WILOR_THIRDPARTY', SYNC_ROOT.parents[2] / 'thirdparty')).resolve()
    model_files = ('AnyHand/pretrained_models/anyhand_wilor.ckpt', 'AnyHand/pretrained_models/model_config_wilor.yaml',
                   'AnyHand/pretrained_models/detector.pt', 'AnyHand/mano_data/MANO_RIGHT.pkl',
                   'da3/weights/DA3-GIANT/config.json', 'da3/weights/DA3-GIANT/model.safetensors',
                   'sam3/checkpoints/sam3.pt', 'Dyn-HaMR/_DATA/hmp_model/results/model/local_encoder.pth',
                   'Dyn-HaMR/_DATA/hmp_model/results/model/global_encoder.pth',
                   'Dyn-HaMR/_DATA/hmp_model/results/model/nemf.pth')
    model_ids = {name: identity(thirdparty / name) if (thirdparty / name).is_file() else None for name in model_files}
    code_hash = hashlib.sha256()
    for p in sorted([*SYNC_ROOT.rglob('*.py'), *SYNC_ROOT.rglob('*.yaml')]):
        code_hash.update(str(p.relative_to(SYNC_ROOT)).encode())
        code_hash.update(p.read_bytes())
    data = {'files': files, 'models': model_ids, 'code': code_hash.hexdigest(), 'camera_mode': camera_mode, 'path': 'b',
            'confidence': confidence, 'backend': 'ego2robot_sync_v1'}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def convert_output(video, work):
    """Expose the original 3D/2D tracks using EgoWhale's established arrays."""
    info = json.loads((work / 'input.json').read_text())
    count, fps = info['frames'], float(info['fps'])
    width, height = info['size']
    camera = json.loads((work / 'cam_intrinsics.json').read_text())['left']
    cap = cv2.VideoCapture(str(video))
    try:
        original_w, original_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if min(original_w, original_h) <= 0:
        raise ValueError('无法读取原视频尺寸')
    sx, sy = original_w / width, original_h / height
    k = np.array([[camera['fx'] * sx, 0, camera['cx'] * sx],
                  [0, camera['fy'] * sy, camera['cy'] * sy], [0, 0, 1]], dtype=np.float64)
    xyz = np.full((count, 2, 21, 3), np.nan, dtype=np.float32)
    uv = np.full((count, 2, 21, 2), np.nan, dtype=np.float32)
    valid = np.zeros((count, 2), dtype=bool)
    for i, side in enumerate(('left', 'right')):
        with np.load(work / 'hand_processor' / f'hand_data_{side}.npz', allow_pickle=True) as hand:
            flag = np.asarray(hand['hand_detected'], dtype=bool)
            if flag.shape != (count,):
                raise ValueError(f'{side} 手部帧数不匹配')
            valid[:, i] = flag
            xyz[flag, i] = hand['kpts_3d'][flag]
            uv[flag, i] = hand['kpts_2d'][flag] * np.array([sx, sy])
    return {'joints_camera': xyz, 'keypoints_2d': uv, 'valid': valid,
            # The reference final files do not contain per-frame confidence or boxes.
            'confidence': np.full((count, 2), np.nan, dtype=np.float32),
            'bbox': np.full((count, 2, 4), np.nan, dtype=np.float32),
            'intrinsic': k, 'frame_index': np.arange(count),
            'timestamps': np.arange(count) / fps, 'fps': np.float32(fps)}


def run(src, dst, config, python, camera_mode, calibration, confidence, intrinsics_json=None, runner=None):
    from egowhale.wilor.step import file_identity, save_prediction
    src, dst = require_mp4(src), Path(dst).resolve()
    video = src
    if calibration or intrinsics_json or camera_mode not in ('auto', 'wilor-default', 'da3'):
        raise ValueError('MP4 输入固定使用 Path B 和 DA3；不能指定其他相机参数')
    work = dst / 'action/wilor_ego2robot'
    work.mkdir(parents=True, exist_ok=True)
    run_signature = signature(src, config, calibration, camera_mode, confidence, intrinsics_json)
    marker_file = work / 'egowhale_run_identity.json'
    if marker_file.exists():
        if json.loads(marker_file.read_text()).get('signature') != run_signature:
            raise ValueError('输入或配置已变化，请使用新的输出目录')
    else:
        if any(work.iterdir()):
            raise ValueError('输出目录含未识别的中间结果，请使用新的输出目录')
        marker_file.write_text(json.dumps({'signature': run_signature}, indent=2))
    if confidence != .3:
        raise ValueError('请在同步配置 YAML 中设置 hand_det_thresh；不支持旧 confidence 覆盖')
    cmd = [str(python), '-u', str(RUNNER), '--video', str(src), '--output', str(work),
           '--config', str(config), '--path', 'b', '--camera-source', 'da3', '--video-only']
    env = os.environ.copy()
    thirdparty = Path(env.get('EGOWHALE_WILOR_THIRDPARTY', SYNC_ROOT.parents[2] / 'thirdparty')).resolve()
    paths = (SYNC_ROOT, thirdparty / 'da3/src', thirdparty / 'sam3', thirdparty / 'human_body_prior/src')
    env['PYTHONPATH'] = os.pathsep.join([*(str(path) for path in paths), env.get('PYTHONPATH', '')])
    if runner is None:
        subprocess.run(cmd, env=env, cwd=SYNC_ROOT, check=True)
    else:
        runner(cmd[3:])
    result = convert_output(video, work)
    marker = json.loads((work / 'full_completed.json').read_text())
    metadata = {'signature': run_signature,
                'input': file_identity(video), 'frames': len(result['valid']),
                'calibration_input': None,
                'intrinsics_json_input': file_identity(Path(intrinsics_json)) if intrinsics_json else None,
                'options': {'camera_mode': camera_mode, 'calibration': str(calibration) if calibration else None,
                            'confidence': confidence, 'intrinsics_json': str(intrinsics_json) if intrinsics_json else None},
                'source': 'ego2robot_sync', 'processing_path': marker.get('path'), 'optimization': marker['optimization'],
                'camera_source': marker['camera_source'], 'overlay': str(work / 'optimized_hand_points.mp4'),
                'confidence_bbox': 'not provided by reference final output',
                'keypoints_2d_coordinates': 'original input resolution',
                'joint_order': 'wrist,thumb,index,middle,ring,little'}
    save_prediction(dst, result, metadata)
