"""DA3 worker: only RGB and explicitly supplied camera inputs enter inference."""
from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

sys.path = [p for p in sys.path if Path(p).resolve() != Path(__file__).resolve().parent]
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))


def infer(request):
    sys.path.insert(0, str(Path.cwd() / 'src'))
    import numpy as np
    import torch
    from depth_anything_3.api import DepthAnything3

    from egowhale.media import read_frames, video_shape
    from egowhale.wilor.pipeline.camera import restore_intrinsic, stitch_camera_windows, summarize_intrinsics
    from egowhale.wilor.pipeline.contracts import Artifact, write_artifact
    from egowhale.wilor.pipeline.progress import report

    options = request['options']
    video = Path(request['video'])
    count, height, width, fps = video_shape(video)
    model = DepthAnything3.from_pretrained(options['checkpoint']).to('cuda:0').eval()
    known = options.get('known_intrinsic')
    motion = options.get('motion', False)
    length = options.get('window', 16)
    overlap = options.get('overlap', 8)
    if not 0 < overlap < length:
        raise ValueError('camera overlap must be between zero and window')
    groups = [np.unique(np.linspace(0, count - 1, min(count, options.get('samples', 16))).astype(int))]
    if motion:
        groups = [np.arange(start, min(start + length, count)) for start in range(0, max(1, count - overlap), length - overlap)]
    windows, all_k, all_ids, all_a = [], [], [], []
    print(f'[DA3] 加载模型完成；采样 {sum(len(ids) for ids in groups)} 帧；'
          f'已知内参输入：{"无" if known is None else "有"}', file=sys.stderr, flush=True)
    for group_index, ids in enumerate(groups):
        images = list(read_frames(video, ids))
        # Obtain A from the exact upstream preprocessing with identity matrices. These are never supplied to inference.
        _, _, transforms = model.input_processor(images, intrinsics=np.tile(np.eye(3), (len(ids), 1, 1)),
            process_res=options.get('process_res', 336), process_res_method='upper_bound_resize')
        transforms = np.asarray(transforms)
        with torch.inference_mode():
            result = model.inference(images, intrinsics=None if known is None else np.tile(known, (len(ids), 1, 1)),
                                     process_res=options.get('process_res', 336), process_res_method='upper_bound_resize')
        ks = restore_intrinsic(result.intrinsics, transforms)
        ext = np.tile(np.eye(4), (len(ids), 1, 1))
        ext[:, :3] = result.extrinsics
        windows.append(Artifact({'w2c': ext, 'intrinsic': ks[0], 'frame_index': ids}, {'kind': 'camera'}))
        all_k.extend(ks)
        all_ids.extend(ids)
        all_a.extend(transforms)
        report('DA3 内参估计批次', group_index + 1, len(groups))
    diagnostics = Path(request['out']).parent / 'da3_windows.npz'
    np.savez_compressed(diagnostics, **{f'w2c_{i}': w.arrays['w2c'] for i, w in enumerate(windows)},
                        **{f'ids_{i}': w.arrays['frame_index'] for i, w in enumerate(windows)})
    samples = np.asarray(all_k)
    k = np.asarray(known) if known is not None else summarize_intrinsics(samples, width, height, options.get('focal_tolerance', .1))
    camera = stitch_camera_windows(windows, count) if motion else Artifact({}, {'kind': 'camera', 'pose_scale': 'unavailable'})
    camera.arrays.update(intrinsic=k, sampled_intrinsics=samples, inference_frame_index=np.asarray(all_ids),
                         image_from_original=np.asarray(all_a), frame_index=np.arange(count), fps=np.array(fps))
    camera.meta.update(width=width, height=height, source='da3', fixed_lens_assumption=True,
                       reference_frame=0, pose_scale='arbitrary' if motion else 'unavailable')
    path = Path(request['out'])
    write_artifact(path, camera)
    return {'ok': True, 'artifact': str(path)}


def main():
    protocol = sys.stdout
    for line in sys.stdin:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                response = infer(json.loads(line))
        except Exception as error:
            import traceback
            traceback.print_exc(file=sys.stderr)
            response = {'ok': False, 'error': f'{type(error).__name__}: {error}'}
        protocol.write(json.dumps(response) + '\n')
        protocol.flush()


if __name__ == '__main__':
    from egowhale.wilor.step import protect_worker_parent
    protect_worker_parent()
    main()
