"""Run with WiLoR's Python; stdout is reserved for the JSON line protocol."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

# This script is executed directly in the third-party environment, without installing EgoWhale there.
sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path.cwd()))


def main():
    protocol = sys.stdout
    model = detector = cfg = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                import cv2
                import numpy as np
                import torch

                from egowhale.wilor.compat import legacy_mano, load_detector

                legacy_mano()

                from egowhale.wilor.camera import crop_camera_to_full, resolve_camera
                from egowhale.wilor.step import pack_detections, save_prediction

                if request.get('operation') == 'mano_forward':
                    response = mano_forward(request)
                    protocol.write(json.dumps(response) + '\n')
                    protocol.flush()
                    continue
                if model is None:
                    from wilor.models import load_wilor
                    if request.get('operation') in ('infer_candidates', 'reconstruct_candidates'):
                        model, cfg = load_candidate_model(request)
                    else:
                        model, cfg = load_wilor(request.get('checkpoint', './pretrained_models/wilor_final.ckpt'), request.get('model_config', './pretrained_models/model_config.yaml'))
                    model = model.to('cuda:0').eval()
                    detector = load_detector(request.get('detector', './pretrained_models/detector.pt'), 'cuda:0')
                if request.get('operation') in ('infer_candidates', 'reconstruct_candidates'):
                    response = infer_candidates(request, model, detector, cfg)
                    protocol.write(json.dumps(response) + '\n')
                    protocol.flush()
                    continue
                from wilor.datasets.vitdet_dataset import ViTDetDataset
                from wilor.utils import recursive_to

                capture = cv2.VideoCapture(request['video'])
                if not capture.isOpened():
                    raise ValueError('cannot open video')
                fps = float(capture.get(cv2.CAP_PROP_FPS))
                width, height = int(capture.get(3)), int(capture.get(4))
                if not np.isfinite(fps) or fps <= 0:
                    capture.release()
                    raise ValueError('video has invalid fps')
                k, camera_meta = resolve_camera(width, height, request['camera_mode'],
                    focal_ratio=cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE,
                    calibration=Path(request['calibration']) if request['calibration'] else None)
                expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                frames = []
                try:
                    while True:
                        ok, image = capture.read()
                        if not ok:
                            break
                        detections = detector(image, conf=request['confidence'], verbose=False)[0].boxes
                        boxes = detections.xyxy.detach().cpu().numpy()
                        right = detections.cls.detach().cpu().numpy()
                        scores = detections.conf.detach().cpu().numpy()
                        rows = []
                        if len(boxes):
                            dataset = ViTDetDataset(cfg, image, boxes, right, rescale_factor=2., fp16=False)
                            loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
                            offset = 0
                            for batch in loader:
                                batch = recursive_to(batch, 'cuda:0')
                                with torch.inference_mode():
                                    prediction = model(batch)
                                cam = prediction['pred_cam'].detach().cpu().numpy().copy()
                                sides = batch['right'].detach().cpu().numpy()
                                multiplier = 2 * sides - 1
                                cam[:, 1] *= multiplier
                                translation = crop_camera_to_full(cam,
                                    batch['box_center'].float().cpu().numpy(), batch['box_size'].float().cpu().numpy(), k)
                                points = prediction['pred_keypoints_3d'].detach().cpu().numpy().copy()
                                points[:, :, 0] *= multiplier[:, None]
                                points += translation[:, None]
                                for n in range(len(points)):
                                    rows.append({'right': bool(sides[n]), 'confidence': float(scores[offset+n]),
                                                 'bbox': boxes[offset+n], 'joints': points[n]})
                                offset += len(points)
                        selected = {}
                        for row in rows:
                            side = row['right']
                            if side not in selected or row['confidence'] > selected[side]['confidence']:
                                selected[side] = row
                        frames.append(list(selected.values()))
                        if len(frames) % 100 == 0:
                            print(f'WiLoR: {len(frames)} frames', file=sys.stderr, flush=True)
                finally:
                    capture.release()
                if not frames:
                    raise ValueError('video has no decoded frames')
                if expected and expected != len(frames):
                    raise ValueError(f'video decoded {len(frames)} frames, expected {expected}')
                result = pack_detections(frames, k, fps)
                metadata = {'signature': request['signature'], 'input': request['input'],
                            'calibration_input': request['calibration_input'], 'frames': len(frames), 'fps': fps,
                            'camera': camera_meta, 'units': 'meters', 'sides': ['left', 'right'],
                            'joint_order': 'OpenPose21: wrist; thumb,index,middle,ring,little (4 joints each)',
                            'selection': 'highest confidence per detector side per frame',
                            'temporal_optimization': None, 'options': request}
                save_prediction(Path(request['out']), result, metadata)
            response = {'ok': True, 'frames': len(frames)}
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        protocol.write(json.dumps(response) + '\n')
        protocol.flush()




def load_candidate_model(request):
    """Load network weights while keeping the configured licensed MANO template."""
    import torch
    from wilor.configs import get_config
    from wilor.models import WiLoR

    cfg = get_config(request['model_config'], update_cachedir=True)
    cfg.defrost()
    if 'vit' in cfg.MODEL.BACKBONE.TYPE and 'BBOX_SHAPE' not in cfg.MODEL:
        cfg.MODEL.BBOX_SHAPE = [192, 256]
    cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
    mano = Path(request['mano']).absolute()
    cfg.MANO.DATA_DIR = str(mano.parent)
    cfg.MANO.MODEL_PATH = str(mano)
    cfg.MANO.MEAN_PARAMS = str(mano.parent / 'mano_mean_params.npz')
    cfg.freeze()
    model = WiLoR(cfg, init_renderer=False)
    checkpoint = torch.load(request['checkpoint'], map_location='cpu', weights_only=False)
    state = {key: value for key, value in checkpoint['state_dict'].items() if not key.startswith('mano.')}
    incompatible = model.load_state_dict(state, strict=False)
    if any(not key.startswith('mano.') for key in incompatible.missing_keys) or incompatible.unexpected_keys:
        raise ValueError(f'incompatible WiLoR network checkpoint: {incompatible}')
    return model, cfg


def pack_candidates(frames, intrinsic, fps):
    import numpy as np

    from egowhale.wilor.camera import project
    from egowhale.wilor.pipeline.contracts import Artifact

    rows = [row for frame in frames for row in frame]
    shapes = {'joints_camera': (21, 3), 'bbox': (4,), 'global_orient': (3, 3), 'hand_pose': (15, 3, 3),
              'betas': (10,), 'crop_camera': (3,), 'translation': (3,)}
    a = {key: np.asarray([row['joints' if key == 'joints_camera' else key] for row in rows], dtype=np.float32).reshape((-1, *shape))
         for key, shape in shapes.items()}
    a.update(offsets=np.r_[0, np.cumsum([len(frame) for frame in frames])].astype(np.int64),
             frame_index=np.arange(len(frames)), fps=np.array(fps), intrinsic=np.asarray(intrinsic),
             detector_side=np.asarray([row['right'] for row in rows], dtype=np.int8),
             confidence=np.asarray([row['confidence'] for row in rows], dtype=np.float32))
    a['reconstruction_side'] = a['detector_side'].copy()
    a['keypoints_2d'] = project(a['joints_camera'], intrinsic).astype(np.float32)
    a['accepted'] = np.isfinite(a['joints_camera']).all(axis=(1, 2)) & (a['joints_camera'][..., 2] > 0).all(axis=1)
    a['reasons'] = np.zeros(len(rows), np.uint16)
    return Artifact(a, {'kind': 'candidates', 'mano_convention': 'right_canonical; mirror x for left before translation',
                        'frames': len(frames), 'units': 'meters', 'sides': ['left', 'right']})


def select_candidates(candidates, ids=None):
    import numpy as np

    from egowhale.wilor.pipeline.contracts import Artifact

    a = candidates.arrays
    t = len(a['offsets']) - 1
    if ids is None:
        ids = np.full((t, 2), -1, np.int64)
        for frame, (start, end) in enumerate(zip(a['offsets'][:-1], a['offsets'][1:])):
            for side in range(2):
                choices = [i for i in range(start, end) if a['accepted'][i] and a['detector_side'][i] == side]
                if choices:
                    ids[frame, side] = max(choices, key=lambda i: (a['confidence'][i], -i))
    valid = ids >= 0
    result = {'candidate_id': ids.copy(), 'valid': valid.copy(), 'observed': valid.copy(),
              'interpolated': np.zeros((t, 2), bool), 'reasons': np.zeros((t, 2), np.uint16),
              'frame_index': np.arange(t), 'fps': a['fps'].copy(), 'intrinsic': a['intrinsic'].copy()}
    for key in ('joints_camera', 'keypoints_2d', 'bbox', 'global_orient', 'hand_pose', 'betas', 'crop_camera', 'translation', 'confidence'):
        shape = a[key].shape[1:]
        result[key] = np.full((t, 2, *shape), np.nan, np.float32)
        result[key][valid] = a[key][ids[valid]]
    return Artifact(result, {**candidates.meta, 'kind': 'raw', 'selection': 'highest confidence per side per frame'})


def infer_candidates(request, model, detector, cfg):
    import cv2
    import numpy as np
    import torch
    from wilor.datasets.vitdet_dataset import ViTDetDataset
    from wilor.utils import recursive_to

    from egowhale.wilor.camera import crop_camera_to_full
    from egowhale.wilor.pipeline.contracts import read_artifact, write_artifact
    from egowhale.wilor.pipeline.progress import report

    k = np.asarray(request['intrinsic'])
    corrections = {int(key): int(side) for key, side in request.get('corrections', {}).items()}
    original = read_artifact(Path(request['candidates']), expected_kind='candidates') if corrections else None
    cap = cv2.VideoCapture(request['video'])
    if not cap.isOpened():
        raise ValueError('cannot open video')
    fps, expected = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            f = len(frames)
            if original is None:
                detection = detector(image, conf=request.get('confidence', .3), verbose=False)[0].boxes
                boxes, sides, scores = (x.detach().cpu().numpy() for x in (detection.xyxy, detection.cls, detection.conf))
                ids = None
            else:
                start, stop = original.arrays['offsets'][f:f+2]
                ids = [i for i in range(start, stop) if i in corrections]
                boxes = original.arrays['bbox'][ids]
                sides = np.asarray([corrections[i] for i in ids])
                scores = original.arrays['confidence'][ids]
            rows = []
            if len(boxes):
                ds = ViTDetDataset(cfg, image, boxes, sides, rescale_factor=2., fp16=False)
                loader = torch.utils.data.DataLoader(ds, batch_size=request.get('batch_size', 16), shuffle=False, num_workers=0)
                offset = 0
                for batch in loader:
                    batch = recursive_to(batch, 'cuda:0')
                    with torch.inference_mode():
                        pred = model(batch)
                    mano = {key: v.detach().cpu().numpy() for key, v in pred['pred_mano_params'].items()}
                    raw_cam = pred['pred_cam'].detach().cpu().numpy().copy()
                    cam = raw_cam.copy()
                    side = batch['right'].detach().cpu().numpy()
                    multiplier = 2 * side - 1
                    cam[:, 1] *= multiplier
                    trans = crop_camera_to_full(cam, batch['box_center'].cpu().numpy(), batch['box_size'].cpu().numpy(), k)
                    pts = pred['pred_keypoints_3d'].detach().cpu().numpy().copy()
                    pts[:, :, 0] *= multiplier[:, None]
                    pts += trans[:, None]
                    for n in range(len(pts)):
                        rows.append({'right': int(side[n]), 'confidence': float(scores[offset+n]), 'bbox': boxes[offset+n],
                            'joints': pts[n], 'global_orient': mano['global_orient'][n].reshape(3, 3),
                            'hand_pose': mano['hand_pose'][n].reshape(15, 3, 3), 'betas': mano['betas'][n],
                            'crop_camera': raw_cam[n], 'translation': trans[n]})
                    offset += len(pts)
            if original is not None and rows:
                packed = pack_candidates([rows], k, fps)
                for j, i in enumerate(ids):
                    for key in ('joints_camera', 'keypoints_2d', 'global_orient', 'hand_pose', 'betas', 'crop_camera', 'translation', 'reconstruction_side', 'accepted'):
                        original.arrays[key][i] = packed.arrays[key][j]
            frames.append(rows)
            report('WiLoR 左右手纠正重建' if corrections else 'WiLoR 逐帧重建', len(frames), expected)
    finally:
        cap.release()
    if not frames or len(frames) != expected:
        raise ValueError('incomplete video decoding')
    out = original if original is not None else pack_candidates(frames, k, fps)
    write_artifact(Path(request['out']), out)
    return {'ok': True, 'artifact': request['out']}


def mano_forward(request):
    import numpy as np
    import torch
    from wilor.models.mano_wrapper import MANO

    from egowhale.wilor.camera import project
    from egowhale.wilor.pipeline.contracts import read_artifact, write_artifact

    artifact = read_artifact(Path(request['input']), expected_kind='final')
    a = artifact.arrays
    frames, sides = np.where(a['interpolated'] & ~a['valid'])
    if len(frames):
        mano = MANO(model_path=request['mano'], is_rhand=True).cuda()
        with torch.inference_mode():
            output = mano(global_orient=torch.tensor(a['global_orient'][frames, sides, None], device='cuda'),
                          hand_pose=torch.tensor(a['hand_pose'][frames, sides], device='cuda'),
                          betas=torch.tensor(a['betas'][frames, sides], device='cuda'), pose2rot=False)
        local = output.joints.cpu().numpy()
        local[..., 0] *= (2*sides - 1)[:, None]
        translation = a['target_wrist'][frames, sides] - local[:, 0]
        xyz = local + translation[:, None]
        a['joints_camera'][frames, sides] = xyz
        a['keypoints_2d'][frames, sides] = project(xyz, a['intrinsic'])
        a['translation'][frames, sides] = translation
        a['valid'][frames, sides] = np.isfinite(xyz).all((1, 2)) & (xyz[..., 2] > 0).all(1)
    write_artifact(Path(request['out']), artifact)
    return {'ok': True, 'artifact': request['out']}


if __name__ == '__main__':
    from egowhale.wilor.step import protect_worker_parent
    protect_worker_parent()
    main()
