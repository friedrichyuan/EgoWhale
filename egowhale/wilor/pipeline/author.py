"""Execute pinned author algorithms without importing unrelated robot stages."""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import textwrap
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from egowhale.paths import ROOT

REPO = ROOT / 'thirdparty/ego2robot-author'
REVISION = '244a4ee02f02b2e93e94a81f27963597e0815ce6'
SOURCE_SHA256 = '3ff69425bafaeb22bd600dbf8ed96bffd35b08406ffde7e42d3e8cb377e7673f'


def source(repo: Path = REPO) -> str:
    path = repo / 'ego2robot/processors/hand_processor.py'
    if not path.is_file():
        raise ValueError(f'author source missing; clone revision {REVISION} into {repo}')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
        raise ValueError('author source changed; use the pinned revision')
    return data.decode()


def namespace() -> dict:
    return {'np': np, 'os': os, 'json': json, 'Path': Path, 'logger': logging.getLogger(__name__)}


def load_method(name: str):
    tree = ast.parse(source())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'HandProcessor')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    scope = namespace()
    exec(compile(ast.Module(body=[method], type_ignores=[]),
                 str(REPO / 'ego2robot/processors/hand_processor.py'), 'exec'), scope)
    return scope[name]


def run_block(start: str, end: str, scope: dict) -> dict:
    values = {**namespace(), **scope}
    exec(compile_block(source(), start, end), values)
    return values


@lru_cache(maxsize=4)
def compile_block(text: str, start: str, end: str):
    begin = text.index(start)
    block = textwrap.dedent(text[begin:text.index(end, begin)])
    return compile(block, str(REPO / 'ego2robot/processors/hand_processor.py'), 'exec')


def association(a: dict) -> tuple[np.ndarray, int | None]:
    count = len(a['offsets']) - 1
    all_dets = {}
    for frame, (lo, hi) in enumerate(zip(a['offsets'][:-1], a['offsets'][1:])):
        choices = []
        for i in range(lo, hi):
            if not a['accepted'][i] or not np.isfinite(a['keypoints_2d'][i, 0]).all():
                continue
            pixels = a['keypoints_2d'][i]
            choices.append({'hand': i, 'joints_cam': None, 'kp2d': pixels, 'projection_focal': None,
                            'score': a['confidence'][i], 'wrist_u': pixels[0, 0], 'wrist_v': pixels[0, 1]})
        if choices:
            all_dets[frame] = choices
    result = run_block('        # Step 2: Score-based tracking', '        # Step 3: Jump filter',
                       {'all_dets': all_dets, 'T': count})
    ids = np.full((count, 2), -1, np.int64)
    for side, key in enumerate(('left_r', 'right_r')):
        for frame, row in result[key].items():
            ids[frame, side] = row['hand']
    return ids, result['best_t']


def jump_rejections(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    tracks = [{int(f): {'joints_cam': points[f, side]} for f in np.flatnonzero(valid[:, side])} for side in range(2)]
    run_block('        # Step 3: Jump filter', '        # Step 4: Save hand_data_',
              {'left_r': tracks[0], 'right_r': tracks[1]})
    rejected = valid.copy()
    for side, track in enumerate(tracks):
        rejected[list(track), side] = False
    return rejected


def filter_detection(pixels: np.ndarray, mask: np.ndarray, edge: bool) -> int:
    """Execute the author's detection-filter loop with an equivalent projection."""
    height, width = mask.shape
    xy = np.where(np.isfinite(pixels), pixels, -100000.)
    hand = SimpleNamespace(keypoints_3d=np.c_[xy - [width / 2, height / 2], np.ones(21)],
                           cam_t=np.zeros(3), focal_length=1., score=1.)
    scope = {**namespace(), 'hands': [hand], 'mask': mask, 'H': height, 'W': width,
             'n_sam3': 0, 'n_edge': 0, 'frame_dets': []}
    exec(compile_filter(source()), scope)
    return (1 if scope['n_sam3'] else 0) | (2 if edge and scope['n_edge'] else 0)


@lru_cache(maxsize=1)
def compile_filter(text: str):
    tree = ast.parse(text)
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_anyhand_dynhamr_full_pipeline')
    loop = next(node for node in ast.walk(method) if isinstance(node, ast.For)
                and isinstance(node.iter, ast.Name) and node.iter.id == 'hands')
    return compile(ast.Module(body=[loop], type_ignores=[]), str(REPO / 'ego2robot/processors/hand_processor.py'), 'exec')
