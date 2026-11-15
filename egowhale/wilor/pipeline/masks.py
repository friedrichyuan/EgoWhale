"""Independent hand masks, distinct from the visual arm-removal segmentation."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from egowhale.wilor.step import WorkerClient

from .contracts import Artifact, derive, read_artifact


def estimate_masks(video: Path, options: dict) -> Artifact:
    client = WorkerClient([str(Path(options['python']).absolute()), str(Path(__file__).parent / 'workers/sam3.py')],
                          Path(options['repo']), options['timeout'])
    try:
        response = client.request({'video': str(Path(video).resolve()), 'out': str(Path(options['out']).resolve()), 'options': options})
        return read_artifact(Path(response['artifact']), expected_kind='masks')
    finally:
        client.close()


def load_mask(masks, index):
    path = Path(masks.meta['files'][index])
    if 'sha256' in masks.meta and hashlib.sha256(path.read_bytes()).hexdigest() != masks.meta['sha256'][index]:
        raise ValueError('mask content changed')
    with np.load(path) as data:
        shape = tuple(data['shape'])
        return np.unpackbits(data['packed'])[:int(np.prod(shape))].reshape(shape).astype(bool)


def filter_candidates(candidates: Artifact, masks: Artifact | None, options: dict) -> Artifact:
    out = derive(candidates, 'candidates', mask_filter=options)
    if not options.get('enabled'):
        return out
    if masks is None:
        raise ValueError('enabled mask filtering requires masks')
    a = out.arrays
    for frame, (start, end) in enumerate(zip(a['offsets'][:-1], a['offsets'][1:])):
        mask = load_mask(masks, frame)
        height, width = mask.shape
        for i in range(start, end):
            pixels = a['keypoints_2d'][i]
            if options.get('backend') == 'author-video':
                from .author import filter_detection
                reason = filter_detection(pixels, mask, options.get('edge', False))
                if reason:
                    a['accepted'][i] = False
                    a['reasons'][i] |= reason
                continue
            finite = np.isfinite(pixels).all(1)
            xy = np.rint(np.where(np.isfinite(pixels), pixels, -100)).astype(np.int64)
            inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
            hits = np.zeros(21, bool)
            hits[inside] = mask[xy[inside, 1], xy[inside, 0]]
            if hits.mean() < .2:
                a['accepted'][i] = False
                a['reasons'][i] |= 1
            if options.get('edge', False):
                near = ~inside | (pixels[:, 0] < 10) | (pixels[:, 0] >= width - 10) | (pixels[:, 1] < 10) | (pixels[:, 1] >= height - 10)
                if near.mean() >= .7:
                    a['accepted'][i] = False
                    a['reasons'][i] |= 2
    out.meta['mask_signature'] = masks.meta.get('signature')
    return out
