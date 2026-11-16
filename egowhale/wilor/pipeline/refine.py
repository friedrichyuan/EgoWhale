"""File-based Dyn-HaMR adapter; raw inputs survive every optimization failure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from egowhale.wilor.step import WorkerClient

from .contracts import Artifact, artifact_signature, canonical, derive, read_artifact, write_artifact


def trim_boundaries(tracks: Artifact, trim: int = 3) -> Artifact:
    """Official postprocess: invalidate first/last N valid samples of each hand."""
    if isinstance(trim, bool) or not isinstance(trim, int) or trim < 0:
        raise ValueError('boundary_trim must be a nonnegative integer')
    out = derive(tracks, tracks.meta['kind'], boundary_trim=trim)
    marked = np.zeros_like(out.arrays['valid'])
    if trim:
        for side in range(2):
            ids = np.flatnonzero(out.arrays['valid'][:, side])
            if len(ids) > 2 * trim:
                marked[ids[:trim], side] = True
                marked[ids[-trim:], side] = True
    out.arrays['boundary_trimmed'] = marked
    out.arrays['valid'][marked] = False
    out.arrays['reasons'][marked] |= 16
    return out


def ensure_workspace_identity(workspace: Path, identity: dict) -> None:
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / 'attempt_identity.json'
    identity = canonical(identity)
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError('optimizer workspace identity changed; use a new output directory')
    else:
        if (workspace / 'result').exists():
            raise ValueError('optimizer checkpoint has no identity; use a new output directory')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(identity, sort_keys=True, indent=2, allow_nan=False))
        temporary.replace(path)


def export_dynhamr_inputs(tracks: Artifact, camera: Artifact, workspace: Path, options: dict) -> Path:
    workspace = Path(workspace).resolve()
    a, c = tracks.arrays, camera.arrays
    is_static = options.get('camera_mode') == 'author-static'
    if is_static:
        c = {**c, 'w2c': np.tile(np.eye(4), (len(a['valid']), 1, 1))}
    if 'w2c' not in c:
        raise ValueError('DynHaMR requires explicit camera motion; estimate it first')
    arbitrary = not is_static and camera.meta.get('pose_scale') == 'arbitrary'
    opt_scale = False if is_static else options.get('opt_scale', True)
    if arbitrary and not opt_scale:
        raise ValueError('arbitrary camera scale requires scale optimization')
    t = len(a['valid'])
    if c['w2c'].shape != (t, 4, 4):
        raise ValueError('camera timeline differs from tracks')
    k = c['intrinsic']
    directory = workspace / 'cameras/shot-0'
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / 'cameras.npz', w2c=c['w2c'].astype(np.float32), height=camera.meta['height'],
             width=camera.meta['width'], focal=k[0, 0], intrins=np.tile([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], (t, 1)))
    sides = [side for side in range(2) if a['valid'][:, side].sum() >= 2]
    for side in sides:
        folder = workspace / 'track_preds' / f'{side:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        for f in np.flatnonzero(a['valid'][:, side]):
            physical = a['global_orient'][f, side].copy()
            if side == 0:
                mirror = np.diag([-1., 1., 1.])
                physical = mirror @ physical @ mirror
            mano = {'global_orient': Rotation.from_matrix(physical).as_rotvec().tolist(),
                    'body_pose': Rotation.from_matrix(a['hand_pose'][f, side]).as_rotvec().tolist(),
                    'cam_trans': a['translation'][f, side].tolist(), 'betas': a['betas'][f, side].tolist(), 'is_right': side}
            (folder / f'{f:06d}_mano.json').write_text(json.dumps(mano, allow_nan=False))
            points = np.c_[a['keypoints_2d'][f, side], np.full(21, a['confidence'][f, side])]
            (folder / f'{f:06d}_keypoints.json').write_text(json.dumps({'people': [{'pose_keypoints_2d': points.reshape(-1).tolist()}]}, allow_nan=False))
    (workspace / 'shots.json').write_text(json.dumps({f'{f:06d}.jpg': 0 for f in range(t)}))
    manifest = {'frames': t, 'fps': float(a['fps']), 'sides': sides, 'opt_scale': bool(opt_scale),
                'is_static': is_static, 'camera_assumption': 'author static identity poses' if is_static else 'supplied motion',
                'pose_scale': camera.meta.get('pose_scale'), 'depth_bounds': options.get('depth_bounds', [.05, .4]),
                'mano_convention': 'physical global rotation, right-canonical finger rotations; mirror local geometry before global rotation'}
    path = workspace / 'input.json'
    path.write_text(json.dumps(manifest, indent=2))
    return path


def import_dynhamr_output(path: Path, camera: Artifact, frame_count: int) -> Artifact:
    result = read_artifact(Path(path), expected_kind='refined')
    if len(result.arrays['valid']) != frame_count or not result.meta.get('optimizer_executed'):
        raise ValueError('invalid or unoptimized DynHaMR output')
    if not np.allclose(result.arrays['intrinsic'], camera.arrays['intrinsic']):
        raise ValueError('optimizer changed calibration')
    return result


def refine_tracks(tracks: Artifact, camera: Artifact, video: Path, options: dict) -> Artifact:
    if not tracks.arrays['valid'].any():
        return derive(tracks, 'refined', optimizer_executed=False, skipped='empty observations')
    workspace = Path(options['workspace'])
    ensure_workspace_identity(workspace, {'tracks': artifact_signature(tracks.meta, tracks.arrays),
        'camera': artifact_signature(camera.meta, camera.arrays), 'options': options})
    manifest = export_dynhamr_inputs(tracks, camera, workspace, options)
    if not json.loads(manifest.read_text())['sides']:
        raise ValueError('DynHaMR needs at least two observations in one hand track')
    write_artifact(workspace / 'tracks.npz', tracks)
    write_artifact(workspace / 'camera.npz', camera)
    client = WorkerClient([str(Path(options['python']).absolute()), str(Path(__file__).parent / 'workers/dynhamr.py')],
                          Path(options['repo']) / 'dyn-hamr', options['timeout'])
    stage_path = workspace / 'optimizer_stage.json'
    stage_path.unlink(missing_ok=True)
    try:
        try:
            response = client.request({'operation': 'refine', 'video': str(Path(video).resolve()), 'workspace': str(workspace.resolve()),
                                       'options': options})
        except Exception as error:
            if not stage_path.is_file() or json.loads(stage_path.read_text()).get('stage') != 'hmp':
                raise
            response = {'optimizer_fallback': True, 'fallback_stage': 'hmp',
                        'fallback_reason': f'{type(error).__name__}: {error}'}
            print('[回退] HMP worker 失败，继续导出 WiLoR 结果：' + response['fallback_reason'], flush=True)
        if response.get('optimizer_fallback') and response.get('fallback_stage') == 'hmp':
            result = derive(tracks, 'refined', optimizer_executed=False, prior_executed=False,
                            optimizer_stage='filtered', optimizer_fallback=True, fallback_stage='hmp',
                            fallback_reason=response['fallback_reason'], root_smooth_completed=True,
                            result_source='wilor_filtered')
            return trim_boundaries(result, options.get('boundary_trim', 3))
        result = import_dynhamr_output(Path(response['artifact']), camera, len(tracks.arrays['valid']))
        if options.get('run_prior', True) and not result.meta.get('prior_executed'):
            raise ValueError('HMP was requested but optimizer did not publish prior output')
        return trim_boundaries(result, options.get('boundary_trim', 3))
    finally:
        client.close()
