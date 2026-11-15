"""Compare a published WiLoR prediction with held-out EgoDex annotations.

Ground truth is read only here, after inference. Coordinates: camera/OpenCV, meters.
Writes metrics.json, errors.npz, and a synchronized GT/prediction overlay video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

JOINTS = ['Hand', *(f'Thumb{s}' for s in ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip')),
          *(f'{finger}Finger{s}' for finger in ('Index', 'Middle', 'Ring', 'Little')
            for s in ('Knuckle', 'IntermediateBase', 'IntermediateTip', 'Tip'))]
EDGES = [(0, start) for start in (1, 5, 9, 13, 17)] + [
    (i, i + 1) for start in (1, 5, 9, 13, 17) for i in range(start, start + 3)]


def project(xyz, k):
    image = xyz @ k.T
    with np.errstate(divide='ignore', invalid='ignore'):
        return image[..., :2] / image[..., 2:]


def load_ground_truth(path):
    with h5py.File(path) as h:
        cameras = np.asarray(h['transforms/camera'], dtype=float)
        k = np.asarray(h['camera/intrinsic'], dtype=float)
        world = np.stack([np.stack([np.asarray(h['transforms/' + side + name])[:, :3, 3]
                                    for name in JOINTS], axis=1) for side in ('left', 'right')], axis=1)
        confidence = np.stack([np.stack([np.asarray(h['confidences/' + side + name])
                                         for name in JOINTS], axis=1) for side in ('left', 'right')], axis=1)
    # EgoDex stores camera-to-world and joint-to-world transforms.
    points = np.einsum('tji,tskj->tski', cameras[:, :3, :3], world - cameras[:, None, None, :3, 3])
    return points, confidence, k


def stats(values, mask):
    data = np.asarray(values)[mask & np.isfinite(values)]
    return {'count': int(data.size), 'mean': float(data.mean()) if data.size else None,
            'median': float(np.median(data)) if data.size else None,
            'p90': float(np.percentile(data, 90)) if data.size else None}


def compare(prediction, gt, confidence, k, width, height, threshold):
    xyz, valid = prediction['joints_camera'], prediction['valid']
    if xyz.shape != gt.shape or valid.shape != gt.shape[:2]:
        raise ValueError('prediction and GT timeline/shapes differ')
    uv_gt, uv_pred = project(gt, k), project(xyz, k)
    gt_valid = ((confidence >= threshold) & np.isfinite(gt).all(-1) & (gt[..., 2] > 0)
                & (uv_gt[..., 0] >= 0) & (uv_gt[..., 0] < width)
                & (uv_gt[..., 1] >= 0) & (uv_gt[..., 1] < height))
    eligible = (gt_valid.sum(-1) >= 5) & (confidence[:, :, 0] >= threshold)
    paired = eligible & valid
    joints = paired[..., None] & gt_valid & np.isfinite(xyz).all(-1)
    error = np.linalg.norm(xyz - gt, axis=-1) * 1000
    relative = np.linalg.norm((xyz - xyz[:, :, :1]) - (gt - gt[:, :, :1]), axis=-1) * 1000
    pixel = np.linalg.norm(uv_pred - uv_gt, axis=-1)
    wrist = np.linalg.norm(xyz[:, :, 0] - gt[:, :, 0], axis=-1) * 1000
    metrics = {'gt_confidence_threshold': threshold, 'eligible_hand_rule': 'at least 5 confident joints in image; confident wrist',
               'coordinate_frame': 'OpenCV camera; unaligned metric errors; mm and pixels',
               'joint_order': JOINTS, 'frames': len(gt), 'sides': {}}
    for side, name in enumerate(('left', 'right')):
        e, p = eligible[:, side], paired[:, side]
        metrics['sides'][name] = {'eligible_hand_frames': int(e.sum()), 'paired_hand_frames': int(p.sum()),
                                 'coverage': float(p.sum() / e.sum()) if e.any() else None,
                                 'mpjpe_mm': stats(error[:, side], joints[:, side]),
                                 'wrist_relative_mpjpe_mm': stats(relative[:, side], joints[:, side]),
                                 'wrist_error_mm': stats(wrist[:, side], p),
                                 'reprojection_error_px': stats(pixel[:, side], joints[:, side])}
    metrics['overall'] = {'mpjpe_mm': stats(error, joints), 'wrist_relative_mpjpe_mm': stats(relative, joints),
                          'wrist_error_mm': stats(wrist, paired), 'reprojection_error_px': stats(pixel, joints)}
    return metrics, {'absolute_error_mm': error, 'relative_error_mm': relative, 'pixel_error': pixel,
                     'evaluated_joints': joints, 'eligible_hands': eligible, 'paired_hands': paired}, uv_gt, uv_pred, gt_valid


def draw_hand(image, points, mask, color):
    for a, b in EDGES:
        if mask[a] and mask[b]:
            cv2.line(image, tuple(points[a].astype(int)), tuple(points[b].astype(int)), color, 2, cv2.LINE_AA)
    for xy in points[mask]:
        cv2.circle(image, tuple(xy.astype(int)), 3, color, -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('prediction', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--confidence', type=float, default=.5)
    args = parser.parse_args()
    with np.load(args.prediction, allow_pickle=False) as data:
        pred = {key: np.asarray(data[key]) for key in data.files}
    gt, confidence, k = load_ground_truth(args.episode)
    if not np.array_equal(pred['frame_index'], np.arange(len(gt))):
        raise ValueError('prediction frame indices are not aligned to annotation')
    if not np.allclose(pred['intrinsic'], k):
        raise ValueError('calibrated comparison requires the same camera intrinsics')
    cap = cv2.VideoCapture(str(args.episode.with_suffix('.mp4')))
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or abs(float(pred['fps']) - fps) > .01:
        raise ValueError('video cannot be opened or prediction FPS differs')
    metrics, errors, uv_gt, uv_pred, visible = compare(pred, gt, confidence, k, width, height, args.confidence)
    meta_path = args.prediction.with_suffix('.json')
    metadata = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    metrics.update(episode=str(args.episode.resolve()), prediction=str(args.prediction.resolve()),
                   evaluation_scope='single-episode comparison; see prediction metadata for enabled processing',
                   published_stage=metadata.get('published_stage', 'legacy'), fps=fps)
    args.out.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out / 'comparison.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('cannot open video writer')
    frame = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if frame >= len(gt):
                raise ValueError('video longer than annotations')
            for side in range(2):
                draw_hand(image, uv_gt[frame, side], visible[frame, side], (0, 220, 0))
                pixels = uv_pred[frame, side]
                mask = np.isfinite(pixels).all(-1) & (np.abs(pixels).max(-1) < 10000) & bool(pred['valid'][frame, side])
                draw_hand(image, pixels, mask, (0, 0, 255))
            cv2.putText(image, f'Frame {frame} | GREEN: EgoDex GT | RED: WiLoR prediction', (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
            writer.write(image)
            frame += 1
    finally:
        cap.release()
        writer.release()
    if frame != len(gt):
        raise ValueError('incomplete video decoding')
    np.savez_compressed(args.out / 'errors.npz', **errors)
    (args.out / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metrics['overall'], indent=2))


if __name__ == '__main__':
    main()
