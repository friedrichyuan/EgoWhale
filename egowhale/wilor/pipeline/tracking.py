"""Paper-style bidirectional wrist association and explicit validity reasons."""
from __future__ import annotations

import numpy as np

from .contracts import Artifact, derive


def associate_candidates(candidates: Artifact, options: dict) -> Artifact:
    a = candidates.arrays
    if options.get('method') == 'author':
        from .author import REVISION, association
        ids, seed = association(a)
        reconstruction = np.zeros(ids.shape, bool)
        for f, side in zip(*np.where(ids >= 0)):
            reconstruction[f, side] = a['reconstruction_side'][ids[f, side]] != side
        return Artifact({'candidate_id': ids, 'needs_reconstruction': reconstruction},
                        {'kind': 'association', 'seed_frame': seed, 'author_revision': REVISION,
                         'strategy': 'author bidirectional tracking; two-hand seed required'})
    t = len(a['offsets']) - 1
    ids = np.full((t, 2), -1, np.int64)
    frames = [[i for i in range(a['offsets'][f], a['offsets'][f+1])
               if a['accepted'][i] and np.isfinite(a['keypoints_2d'][i, 0]).all()] for f in range(t)]
    seeds = []
    for f, choices in enumerate(frames):
        best = [max((i for i in choices if a['detector_side'][i] == side),
                    key=lambda i: (a['confidence'][i], -i), default=-1) for side in range(2)]
        if any(i >= 0 for i in best):
            seeds.append((all(i >= 0 for i in best), sum(a['confidence'][i] for i in best if i >= 0), -f, best))
    seed = None
    if seeds:
        _, _, negative, best = max(seeds, key=lambda row: row[:3])
        seed = -negative
        ids[seed] = best
        for direction in (-1, 1):
            previous = {side: a['keypoints_2d'][i, 0].copy() for side, i in enumerate(best) if i >= 0}
            for frame in range(seed + direction, t if direction == 1 else -1, direction):
                assigned = {side: [] for side in previous}
                for i in frames[frame]:
                    wrist = a['keypoints_2d'][i, 0]
                    side = min(previous, key=lambda side: (np.linalg.norm(wrist - previous[side]),
                                                           side != a['detector_side'][i], side))
                    assigned[side].append(i)
                for side, choices in assigned.items():
                    if choices:
                        i = max(choices, key=lambda i: (a['confidence'][i], -i))
                        ids[frame, side] = i
                        previous[side] = a['keypoints_2d'][i, 0].copy()
    reconstruction = np.zeros((t, 2), bool)
    for f, side in zip(*np.where(ids >= 0)):
        reconstruction[f, side] = a['reconstruction_side'][ids[f, side]] != side
    return Artifact({'candidate_id': ids, 'needs_reconstruction': reconstruction},
                    {'kind': 'association', 'seed_frame': seed, 'strategy': 'bidirectional nearest image wrist',
                     'identity_limit': 'crossing hands may remain ambiguous'})


def reject_jumps(tracks: Artifact, options: dict) -> Artifact:
    out = derive(tracks, tracks.meta['kind'], jump_filter=options)
    if not options.get('enabled', True):
        return out
    a = out.arrays
    if options.get('method') == 'author':
        from .author import REVISION, jump_rejections
        bad = jump_rejections(a['joints_camera'], a['valid'])
        a['valid'][bad] = False
        a['reasons'][bad] |= 4
        out.meta['author_revision'] = REVISION
        return out
    fps = float(a['fps'])
    thresholds = []
    for side in range(2):
        ids = np.flatnonzero(a['valid'][:, side])
        if len(ids) < 2:
            thresholds.append(None)
            continue
        points = a['joints_camera'][ids, side, 0]
        speed = np.linalg.norm(np.diff(points, axis=0), axis=1) / (np.diff(ids) / fps)
        threshold = max(4 * float(np.median(speed)), .003 * 30)
        thresholds.append(threshold)
        bad = ids[1:][speed > threshold]
        a['valid'][bad, side] = False
        a['reasons'][bad, side] |= 4
    out.meta['jump_threshold_m_per_second'] = thresholds
    return out


def apply_depth_bounds(tracks: Artifact, z_min: float, z_max: float) -> Artifact:
    if not 0 < z_min < z_max:
        raise ValueError('invalid depth bounds')
    out = derive(tracks, tracks.meta['kind'], depth_bounds=[z_min, z_max])
    a = out.arrays
    depth = a['joints_camera'][:, :, 0, 2]
    bad = a['valid'] & (~np.isfinite(depth) | (depth < z_min) | (depth > z_max))
    a['valid'][bad] = False
    a['reasons'][bad] |= 8
    return out
