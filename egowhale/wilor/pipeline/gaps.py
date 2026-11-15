"""Hand-only short-gap interpolation, with robot-home requests left to actions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from egowhale.wilor.step import WorkerClient

from .contracts import Artifact, derive, read_artifact, write_artifact


def interpolate_parameters(tracks: Artifact, options: dict) -> tuple[Artifact, list[dict]]:
    out = derive(tracks, 'final', gap_strategy='MANO interpolation; robot-home requests only')
    a = out.arrays
    count = len(a['valid'])
    maximum = int(np.floor(float(a['fps']) * options.get('max_gap_seconds', 10 / 30) + 1e-8))
    a['target_wrist'] = a['joints_camera'][:, :, 0].copy()
    requests = []
    for side in range(2):
        valid = a['valid'][:, side].copy()
        if not valid.any():
            continue
        missing = np.flatnonzero(~valid)
        for run in np.split(missing, np.flatnonzero(np.diff(missing) > 1) + 1):
            if not len(run):
                continue
            start, end = int(run[0]), int(run[-1] + 1)
            if 'boundary_trimmed' in a and a['boundary_trimmed'][run, side].any():
                continue
            if len(run) > maximum:
                requests.append({'side': side, 'start': start, 'end': end, 'reason': 'long_gap'})
                continue
            if start == 0 or end == count:
                continue
            if 'segment_id' in a and a['segment_id'][start - 1] != a['segment_id'][end]:
                continue
            lo, hi = start - 1, end
            bounds = options.get('depth_bounds')
            if bounds and not all(bounds[0] <= a['target_wrist'][f, side, 2] <= bounds[1] for f in (lo, hi)):
                continue
            alpha = (run - lo) / (hi - lo)
            for key in ('betas', 'target_wrist'):
                a[key][run, side] = (1 - alpha[:, None]) * a[key][lo, side] + alpha[:, None] * a[key][hi, side]
            for key in ('global_orient', 'hand_pose'):
                rotations = a[key][[lo, hi], side].reshape(2, -1, 3, 3)
                interpolated = np.stack([Slerp([lo, hi], Rotation.from_matrix(rotations[:, j]))(run).as_matrix()
                                         for j in range(rotations.shape[1])], axis=1)
                a[key][run, side] = interpolated.reshape((len(run), *a[key].shape[2:]))
            if 'confidence' in a:
                a['confidence'][run, side] = np.minimum(a['confidence'][lo, side], a['confidence'][hi, side])
            a['interpolated'][run, side] = True
            a['observed'][run, side] = False
            # valid becomes true only after MANO reconstruction and numerical checks.
    out.meta['home_requests'] = requests
    return out, requests


def fill_hand_gaps(tracks: Artifact, options: dict) -> Artifact:
    out, _ = interpolate_parameters(tracks, options)
    if not out.arrays['interpolated'].any():
        return out
    path = Path(options['out'])
    write_artifact(path.with_name('gap_parameters.npz'), out)
    client = WorkerClient([str(Path(options['python']).absolute()), str(Path(__file__).parents[1] / 'worker.py')],
                          Path(options['repo']), options['timeout'])
    try:
        response = client.request({'operation': 'mano_forward', 'input': str(path.with_name('gap_parameters.npz').resolve()),
                                   'out': str(path.resolve()), 'mano': str(Path(options['mano']).resolve())})
        return read_artifact(Path(response['artifact']), expected_kind='final')
    finally:
        client.close()
