"""WiLoR-owned optional stages; no dependency on the global batch DAG."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from egowhale.media import video_path
from egowhale.paths import GRIPPER, HANDS_WILOR, HANDS_WILOR_META, ROOT
from egowhale.wilor.step import WorkerClient, file_identity

from .contracts import Artifact, canonical, derive, read_artifact, write_artifact

BASE = ROOT / 'thirdparty'
DEFAULTS = {
    'camera': {'source': 'wilor-default', 'calibration': None, 'poses': None, 'pose_scale': 'arbitrary', 'motion': False,
               'python': str(BASE / '.venvs/wilor-vision/bin/python'), 'repo': str(BASE / 'da3'),
               'checkpoint': str(BASE / 'da3/weights/DA3-GIANT'), 'timeout': 1800, 'samples': 16,
               'window': 16, 'overlap': 8, 'process_res': 336, 'focal_tolerance': .1},
    'wilor': {'python': str(BASE / 'WiLoR/.venv/bin/python'), 'repo': str(BASE / 'WiLoR'),
              'checkpoint': str(BASE / 'WiLoR/pretrained_models/wilor_final.ckpt'),
              'detector': str(BASE / 'WiLoR/pretrained_models/detector.pt'),
              'model_config': str(BASE / 'WiLoR/pretrained_models/model_config.yaml'),
              'mano': str(BASE / 'WiLoR/mano_data/MANO_RIGHT.pkl'), 'confidence': .3, 'timeout': 1800,
              'batch_size': 16},
    'mask': {'enabled': False, 'edge': False, 'confidence': .5, 'python': str(BASE / '.venvs/wilor-vision/bin/python'),
             'repo': str(BASE / 'sam3'), 'checkpoint': str(BASE / 'sam3/checkpoints/sam3.pt'), 'timeout': 3600,
             'backend': 'image', 'max_video_gb': 4},
    'tracking': {'enabled': False, 'method': 'native'}, 'jump': {'enabled': False, 'method': 'native'},
    'refine': {'enabled': False, 'python': str(BASE / '.venvs/wilor-dynhamr/bin/python'), 'repo': str(BASE / 'Dyn-HaMR'),
               'timeout': 7200, 'opt_scale': True, 'root_iters': 50, 'smooth_iters': 300, 'lbfgs_max_iter': 20,
               'camera_mode': 'moving', 'run_prior': True, 'boundary_trim': 3,
               'prior_config': str(BASE / 'Dyn-HaMR/dyn-hamr/HMP/hmp_config.yaml')},
    'depth_bounds': {'enabled': True, 'min': .05, 'max': .4},
    'gaps': {'enabled': False, 'max_gap_seconds': 10 / 30}, 'publish': {'stage': 'raw'},
}


def load_options(path: Path) -> dict:
    obj = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(obj, dict):
        raise ValueError('pipeline config must be a mapping')
    out = deepcopy(DEFAULTS)
    if set(obj) - set(out):
        raise ValueError(f'unknown configuration sections: {set(obj) - set(out)}')
    for section, values in obj.items():
        if not isinstance(values, dict) or set(values) - set(out[section]):
            raise ValueError(f'unknown configuration keys in {section}')
        out[section].update(values)
    for values in out.values():
        for key in ('python', 'repo', 'checkpoint', 'detector', 'model_config', 'mano', 'calibration', 'poses', 'prior_config'):
            if values.get(key) is not None:
                p = Path(values[key]).expanduser()
                values[key] = str((ROOT / p).absolute() if not p.is_absolute() else p.absolute())
        if 'enabled' in values and not isinstance(values['enabled'], bool):
            raise ValueError('enabled must be a boolean')
        if 'timeout' in values and values['timeout'] <= 0:
            raise ValueError('timeout must be positive')
    if out['camera']['source'] not in ('da3', 'provided', 'egodex', 'wilor-default'):
        raise ValueError('unknown camera source')
    if out['camera']['pose_scale'] not in ('metric', 'arbitrary'):
        raise ValueError('unknown camera pose_scale')
    if out['mask']['backend'] not in ('image', 'author-video') or out['mask']['max_video_gb'] <= 0:
        raise ValueError('invalid mask backend/memory limit')
    if any(out[key]['method'] not in ('native', 'author') for key in ('tracking', 'jump')):
        raise ValueError('unknown tracking/jump method')
    if out['refine']['camera_mode'] not in ('moving', 'author-static'):
        raise ValueError('unknown refine camera_mode')
    if not isinstance(out['refine']['run_prior'], bool):
        raise ValueError('refine.run_prior must be a boolean')
    trim = out['refine']['boundary_trim']
    if isinstance(trim, bool) or not isinstance(trim, int) or trim < 0:
        raise ValueError('refine.boundary_trim must be a nonnegative integer')
    if out['refine']['enabled'] and out['refine']['camera_mode'] == 'author-static' and (out['camera']['poses'] or out['camera']['motion']):
        raise ValueError('author-static cannot consume explicit camera motion/poses')
    if not 0 < out['camera']['overlap'] < out['camera']['window'] or out['camera']['samples'] < 1:
        raise ValueError('invalid camera sampling/window')
    if out['camera']['focal_tolerance'] < 0 or out['camera']['process_res'] < 28:
        raise ValueError('invalid camera reliability/resolution')
    if out['publish']['stage'] not in ('raw', 'filtered', 'refined', 'final'):
        raise ValueError('unknown publish stage')
    if not 0 < out['wilor']['confidence'] < 1 or not 0 < out['mask']['confidence'] < 1:
        raise ValueError('confidence must be in (0,1)')
    batch = out['wilor']['batch_size']
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError('wilor.batch_size must be a positive integer')
    if not 0 < out['depth_bounds']['min'] < out['depth_bounds']['max']:
        raise ValueError('invalid depth bounds')
    if min(out['refine']['root_iters'], out['refine']['smooth_iters'], out['refine']['lbfgs_max_iter']) < 1:
        raise ValueError('optimizer iterations must be positive')
    if out['gaps']['max_gap_seconds'] < 0:
        raise ValueError('gap duration cannot be negative')
    return out


@lru_cache(maxsize=128)
def _hash_file(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def content_identity(path: str | Path) -> dict:
    identity = file_identity(Path(path))
    return {**identity, 'sha256': _hash_file(identity['path'], identity['size'], identity['mtime_ns'])}


def implementation_identity(name: str) -> dict:
    files = {
        'camera': ['camera.py', 'workers/da3.py'], 'candidates': ['../worker.py'],
        'raw': ['../worker.py'], 'masks': ['masks.py', 'workers/sam3.py'],
        'filtered': ['masks.py', 'tracking.py', '../worker.py'],
        'refined': ['refine.py', 'hmp.py', 'workers/dynhamr.py', 'tracking.py'],
        'final': ['gaps.py', '../worker.py', 'tracking.py'],
    }[name] + ['contracts.py', 'runner.py', '../step.py', '../compat.py', 'author.py']
    return {file: content_identity(Path(__file__).parent / file)['sha256'] for file in files}


def model_identity(section: str, options: dict) -> dict:
    if section == 'author':
        from .author import REPO, REVISION, SOURCE_SHA256, source
        source()
        return {'repo': str(REPO), 'revision': REVISION, 'hand_source_sha256': SOURCE_SHA256}
    o = options[section]
    repo = Path(o['repo'])
    revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    files = {'wilor': ['checkpoint', 'detector', 'model_config', 'mano'], 'mask': ['checkpoint'],
             'camera': [], 'refine': []}[section]
    paths = [Path(o[key]) for key in files]
    if section == 'wilor':
        paths.append(Path(o['mano']).parent / 'mano_mean_params.npz')
    if section == 'camera':
        paths = [Path(o['checkpoint']) / name for name in ('model.safetensors', 'config.json')]
    if section == 'mask':
        paths.append(repo / 'sam3/assets/bpe_simple_vocab_16e6.txt.gz')
    prior_issue = None
    if section == 'refine':
        paths = [Path(options['wilor']['mano']), *sorted((repo / '_DATA/BMC').glob('*.npy'))]
        if len(paths) < 8:
            raise FileNotFoundError('DynHaMR BMC assets missing')
        if o.get('run_prior', True):
            from .hmp import resource_paths
            try:
                assets = resource_paths(repo, Path(o['prior_config']))
                absent = [str(p) for p in assets if not p.is_file()]
                if absent:
                    prior_issue = 'HMP assets missing: ' + ', '.join(absent)
                paths.extend(p for p in assets if p.is_file())
            except ValueError as error:
                prior_issue = str(error)
    if not Path(o['python']).is_file():
        raise FileNotFoundError(o['python'])
    identity = {'repo': str(repo), 'revision': revision, 'files': [content_identity(p) for p in paths]}
    if prior_issue:
        identity['prior_resources_issue'] = prior_issue
    return identity


def _run_wilor(video, camera, options, out, *, candidates=None, corrections=None):
    o = options['wilor']
    client = WorkerClient([str(Path(o['python']).absolute()), str(Path(__file__).parents[1] / 'worker.py')],
                          Path(o['repo']), o['timeout'])
    try:
        response = client.request({**o, 'operation': 'reconstruct_candidates' if corrections else 'infer_candidates',
            'video': str(video.resolve()), 'out': str(out.resolve()), 'intrinsic': camera.arrays['intrinsic'].tolist(),
            'candidates': str(candidates.resolve()) if candidates else None, 'corrections': corrections or {}})
        return read_artifact(Path(response['artifact']), expected_kind='candidates')
    finally:
        client.close()


def validate_published(src: Path, dst: Path, options: dict | None = None) -> dict:
    path = Path(dst) / HANDS_WILOR
    artifact = read_artifact(path, expected_kind='prediction')
    meta = artifact.meta
    if meta['input'] != file_identity(video_path(src)) or meta['video_content'] != content_identity(video_path(src)):
        raise ValueError('WiLoR input changed; use a new output directory')
    calibration = meta.get('calibration_input')
    if calibration is not None and file_identity(Path(calibration['path'])) != calibration:
        raise ValueError('WiLoR calibration changed')
    for key, identity in meta.get('camera_inputs', {}).items():
        if content_identity(identity['path']) != identity:
            raise ValueError(f'WiLoR {key} changed')
    if options and options.get('pipeline_config'):
        config = load_options(Path(options['pipeline_config']))
        if config != meta['pipeline_options']:
            raise ValueError('WiLoR pipeline configuration changed')
    for record in meta.get('artifacts', []):
        upstream = read_artifact(Path(record['path']), expected_kind=record['kind'])
        if upstream.meta['signature'] != record['signature']:
            raise ValueError('WiLoR upstream artifact changed')
    return meta


def validate_pipeline_models(meta: dict) -> None:
    options = meta['pipeline_options']
    checked = {}
    for record in meta.get('artifacts', []):
        artifact = read_artifact(Path(record['path']), expected_kind=record['kind'])
        if artifact.meta.get('implementation') != implementation_identity(record['kind']):
            raise ValueError('WiLoR implementation changed; use a new output directory')
        for section, expected in artifact.meta.get('models', {}).items():
            if section not in checked:
                checked[section] = model_identity(section, options)
            if checked[section] != expected:
                raise ValueError('WiLoR model changed; use a new output directory')


def publish_prediction(src: Path, dst: Path, artifact: Artifact, options: dict, *, artifacts=()) -> None:
    o = options['camera']
    result = derive(artifact, 'prediction', pipeline=True, pipeline_options=options, artifacts=list(artifacts),
                    published_stage=artifact.meta['kind'], input=file_identity(video_path(src)),
                    video_content=content_identity(video_path(src)),
                    calibration_input=file_identity(Path(o['calibration'])) if o['calibration'] else None,
                    camera_inputs={key: content_identity(o[key]) for key in ('calibration', 'poses') if o[key]},
                    options={'camera_mode': o['source'], 'calibration': o['calibration'], 'confidence': options['wilor']['confidence']})
    result.arrays['timestamps'] = np.arange(len(result.arrays['valid'])) / float(result.arrays['fps'])
    path = Path(dst) / HANDS_WILOR
    if path.exists() or path.with_suffix('.json').exists():
        existing = read_artifact(path, expected_kind='prediction')
        if existing.meta['upstream_signature'] == artifact.meta['signature'] and existing.meta['pipeline_options'] == options:
            return
        raise ValueError('published selection changed; use a new output directory')
    if (Path(dst) / GRIPPER).exists() or (Path(dst) / 'action/base.json').exists():
        raise ValueError('existing downstream outputs; use a new output directory')
    write_artifact(path, result)


def run_pipeline(src: Path, dst: Path, options: dict, *, stages: tuple[str, ...] | None = None) -> Artifact:
    from egowhale.wilor.worker import select_candidates

    from .camera import estimate_camera
    from .gaps import fill_hand_gaps
    from .masks import estimate_masks, filter_candidates
    from .refine import refine_tracks
    from .tracking import apply_depth_bounds, associate_candidates, reject_jumps

    video = video_path(Path(src)).resolve()
    dst = Path(dst).resolve()
    root = dst / 'action/wilor_pipeline'
    root.mkdir(parents=True, exist_ok=True)
    filtered = any(options[key]['enabled'] for key in ('mask', 'tracking', 'jump'))
    target = 'final' if options['gaps']['enabled'] else 'refined' if options['refine']['enabled'] else 'filtered' if filtered else 'raw'
    requested = stages if stages is not None else (target,)
    if not requested or set(requested) - {'camera', 'raw', 'masks', 'filtered', 'refined', 'final'}:
        raise ValueError('unknown or empty stages')
    identity = {'options': options, 'video': content_identity(video), 'camera_inputs': {
        key: content_identity(options['camera'][key]) for key in ('calibration', 'poses') if options['camera'][key]}}
    run_id = hashlib.sha256(json.dumps(canonical(identity), sort_keys=True).encode()).hexdigest()
    marker = root / 'run.json'
    if marker.exists() and json.loads(marker.read_text())['run_id'] != run_id:
        raise ValueError('pipeline configuration/input changed; use a new output directory')
    if not marker.exists():
        if (dst / HANDS_WILOR).exists() or (dst / HANDS_WILOR_META).exists() or (dst / GRIPPER).exists():
            raise ValueError('different or existing downstream outputs; use a new output directory')
        marker.write_text(json.dumps({'run_id': run_id, **identity}, indent=2))
    made = {}
    records = []

    def get(name):
        if name in made:
            return made[name]
        dependencies = []
        models = {}
        if name == 'camera':
            cam_o = {**options['camera'], 'motion': options['camera']['motion'] or
                     (options['refine']['enabled'] and options['refine']['camera_mode'] == 'moving')}
            if cam_o['source'] == 'da3' or (cam_o['motion'] and not cam_o['poses']):
                models['camera'] = model_identity('camera', options)
        elif name in ('candidates', 'raw'):
            dependencies = [get('camera')] if name == 'candidates' else [get('candidates')]
            if name == 'candidates':
                models['wilor'] = model_identity('wilor', options)
        elif name == 'masks':
            if not options['mask']['enabled']:
                raise ValueError('masks stage is disabled')
            models['mask'] = model_identity('mask', options)
        elif name == 'filtered':
            dependencies = [get('candidates'), get('raw')]
            if options['mask']['enabled']:
                dependencies.append(get('masks'))
        elif name == 'refined':
            if not options['refine']['enabled']:
                raise ValueError('refine stage is disabled')
            dependencies = [get('filtered' if filtered else 'raw'), get('camera')]
            if dependencies[0].arrays['valid'].any():
                models['refine'] = model_identity('refine', options)
        elif name == 'final':
            dependencies = [get('refined' if options['refine']['enabled'] else 'filtered' if filtered else 'raw')]
            if options['gaps']['enabled']:
                models['wilor'] = model_identity('wilor', options)
        if ((name == 'masks' and options['mask']['backend'] == 'author-video') or
                (name == 'filtered' and any(options[key]['enabled'] and options[key]['method'] == 'author'
                                           for key in ('tracking', 'jump')))):
            models['author'] = model_identity('author', options)
        implementation = implementation_identity(name)
        signatures = [a.meta['signature'] for a in dependencies]
        path = root / f'{name}.npz'
        if path.exists() or path.with_suffix('.json').exists():
            artifact = read_artifact(path, expected_kind=name)
            if artifact.meta.get('run_id') != run_id or artifact.meta.get('dependency_signatures') != signatures or artifact.meta.get('models') != models or artifact.meta.get('implementation') != implementation:
                raise ValueError(f'{name} upstream/model changed; use a new output directory')
        else:
            print(f'WiLoR pipeline: {name}', file=sys.stderr, flush=True)
            if name == 'camera':
                if cam_o['poses']:
                    cam_o['motion'] = False
                artifact = estimate_camera(video, {**cam_o, 'out': str(root / 'camera_worker.npz')})
                if cam_o['poses']:
                    with np.load(cam_o['poses']) as data:
                        artifact.arrays['w2c'] = np.asarray(data['w2c'])
                    artifact.meta['pose_scale'] = cam_o['pose_scale']
            elif name == 'candidates':
                artifact = _run_wilor(video, dependencies[0], options, root / 'candidates_worker.npz')
            elif name == 'raw':
                artifact = select_candidates(dependencies[0])
            elif name == 'masks':
                artifact = estimate_masks(video, {**options['mask'], 'out': str(root / 'masks_worker.npz')})
            elif name == 'filtered':
                cand = filter_candidates(dependencies[0], dependencies[2] if options['mask']['enabled'] else None, options['mask'])
                if options['tracking']['enabled']:
                    association = associate_candidates(cand, options['tracking'])
                    ids = association.arrays['candidate_id']
                    fixes = {str(ids[f, side]): int(side) for f, side in zip(*np.where(association.arrays['needs_reconstruction']))}
                    if fixes:
                        cand = _run_wilor(video, get('camera'), options, root / 'corrected_candidates.npz', candidates=root / 'candidates.npz', corrections=fixes)
                        cand = filter_candidates(cand, dependencies[2] if options['mask']['enabled'] else None, options['mask'])
                        ids = ids.copy()
                        selected = ids >= 0
                        reject = np.zeros_like(selected)
                        reject[selected] = ~cand.arrays['accepted'][ids[selected]]
                        ids[reject] = -1
                    artifact = select_candidates(cand, ids)
                    artifact.meta['association'] = association.meta
                else:
                    artifact = select_candidates(cand)
                artifact = derive(artifact, 'filtered')
                for frame, (lo, hi) in enumerate(zip(cand.arrays['offsets'][:-1], cand.arrays['offsets'][1:])):
                    for side in range(2):
                        rejected = [i for i in range(lo, hi) if cand.arrays['detector_side'][i] == side and not cand.arrays['accepted'][i]]
                        if not artifact.arrays['valid'][frame, side] and rejected:
                            artifact.arrays['reasons'][frame, side] |= np.bitwise_or.reduce(cand.arrays['reasons'][rejected])
                artifact = reject_jumps(artifact, options['jump'])
            elif name == 'refined':
                artifact = refine_tracks(dependencies[0], dependencies[1], video,
                    {**options['refine'], 'workspace': str(root / 'dynhamr'), 'mano': options['wilor']['mano'],
                     'identity': {'models': models, 'implementation': implementation, 'dependencies': signatures}, 'depth_bounds': [options['depth_bounds']['min'], options['depth_bounds']['max']] if options['depth_bounds']['enabled'] else None})
                if options['depth_bounds']['enabled']:
                    artifact = apply_depth_bounds(artifact, options['depth_bounds']['min'], options['depth_bounds']['max'])
            elif name == 'final':
                if options['gaps']['enabled']:
                    gap_o = {**options['wilor'], **options['gaps'], 'out': str(root / 'final_worker.npz')}
                    if options['refine']['enabled'] and options['depth_bounds']['enabled']:
                        gap_o['depth_bounds'] = [options['depth_bounds']['min'], options['depth_bounds']['max']]
                    artifact = fill_hand_gaps(dependencies[0], gap_o)
                    if 'depth_bounds' in gap_o:
                        artifact = apply_depth_bounds(artifact, *gap_o['depth_bounds'])
                else:
                    artifact = derive(dependencies[0], 'final', gaps_disabled=True)
            artifact.meta.update(kind=name, run_id=run_id, dependency_signatures=signatures, models=models, implementation=implementation)
            write_artifact(path, artifact)
        made[name] = artifact
        records.append({'path': str(path), 'kind': name, 'signature': artifact.meta['signature']})
        return artifact

    result = None
    for name in requested:
        result = get(name)
    if options['publish']['stage'] in made and any(name in requested for name in ('raw', 'filtered', 'refined', 'final')):
        publish = get(options['publish']['stage'])
        publish_prediction(video, dst, publish, options, artifacts=records)
    return result
