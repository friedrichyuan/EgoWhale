"""Adapt the upstream 128-frame HMP prior to an unchanged episode timeline."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

WINDOW = 128
OVERLAP = 16


def resource_paths(repo: Path, config: Path) -> list[Path]:
    """Assets actually loaded by upstream Architecture in inference mode."""
    if Path(config).is_file():
        settings = yaml.safe_load(Path(config).read_text())
        data = settings.get('data', {})
        if (data.get('clip_length') != WINDOW or data.get('fps') != 30
                or data.get('gender') != 'neutral' or not data.get('root_transform')):
            raise ValueError('HMP adapter requires the upstream neutral/128-frame/30fps root-transform prior')
    root = Path(repo) / '_DATA/hmp_model'
    return [Path(config), root / 'mean-neutral-128-30fps.pt', root / 'std-neutral-128-30fps.pt',
            *(root / 'results/model' / name for name in
              ('local_encoder.pth', 'global_encoder.pth', 'nemf.pth'))]


def windows(count: int) -> list[tuple[int, int]]:
    if count < 1:
        raise ValueError('HMP requires a nonempty timeline')
    spans = []
    for start in range(0, count, WINDOW - OVERLAP):
        spans.append((start, min(start + WINDOW, count)))
        if spans[-1][1] == count:
            break
    return spans


def padded_indices(start: int, stop: int) -> np.ndarray:
    return np.minimum(np.arange(start, start + WINDOW), stop - 1)


def select_result(output: Path, run_prior: bool) -> Path:
    stage = 'prior' if run_prior else 'smooth_fit'
    paths = sorted((Path(output) / stage).glob('*_world_results.npz'))
    if not paths:
        raise RuntimeError(f'optimizer did not produce {stage} result')
    return paths[-1]


def merge_windows(base: dict, chunks: list[tuple[int, int, int, dict]]) -> dict:
    """Blend overlaps, on SO(3) for rotations, without retaining padded frames."""
    out = {k: v.copy() for k, v in base.items()}
    hands, count = base['trans'].shape[:2]
    weights = np.zeros((hands, count), np.int32)
    beta_sum = np.zeros((hands, 10), np.float64)
    beta_count = np.zeros(hands, np.int32)
    out['pose_body'] = out['pose_body'].reshape(hands, count, 45)
    for side, start, stop, chunk in chunks:
        n = stop - start
        ids = slice(start, stop)
        old = weights[side, ids]
        alpha = 1. / (old + 1)
        for key in ('trans', 'root_orient', 'pose_body'):
            value = np.asarray(chunk[key]).reshape(1, WINDOW, -1)[0, :n]
            target = out[key][side, ids]
            if value.shape != target.shape or not np.isfinite(value).all():
                raise ValueError(f'invalid HMP {key} window')
            if key == 'trans':
                out[key][side, ids] = target * (1 - alpha[:, None]) + value * alpha[:, None]
            else:
                a = Rotation.from_rotvec(target.reshape(-1, 3))
                b = Rotation.from_rotvec(value.reshape(-1, 3))
                fraction = np.repeat(alpha, target.shape[-1] // 3)
                out[key][side, ids] = (a * Rotation.from_rotvec((a.inv() * b).as_rotvec() * fraction[:, None])).as_rotvec().reshape(n, -1)
        if not np.array_equal(np.asarray(chunk['is_right']).reshape(1, WINDOW)[0, :n], base['is_right'][side, ids]):
            raise ValueError('HMP changed handedness')
        beta = np.asarray(chunk['betas']).reshape(-1, 10)
        if not np.isfinite(beta).all():
            raise ValueError('invalid HMP betas')
        beta_sum[side] += beta.mean(0) * n
        beta_count[side] += n
        weights[side, ids] += 1
    if not weights.all():
        raise ValueError('HMP windows do not cover full timeline')
    out['betas'] = (beta_sum / beta_count[:, None]).astype(base['betas'].dtype)
    return out


def run_prior(cfg, dataset, output: Path, device, upstream, options: dict, images: Path) -> Path:
    """Reuse upstream run_prior/fitting/learned prior, one padded hand-window at a time."""
    import torch
    from body_model import MANO
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from util.tensor import move_to

    from .progress import report
    from .workers.dynhamr import physical_run_mano

    fitting = __import__('HMP.fitting', fromlist=['Arguments'])
    original_arguments = fitting.Arguments

    def licensed_arguments(*args, **kwargs):
        result = original_arguments(*args, **kwargs)
        result.smpl.smpl_body_model = str(Path(options['mano']).parent)
        return result

    source = select_result(output, False)
    with np.load(source, allow_pickle=False) as data:
        base = {k: np.asarray(data[k]) for k in data.files}
    hands, count = base['trans'].shape[:2]
    obs = move_to(next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False))), device)
    model = MANO(batch_size=WINDOW, pose2rot=True,
                 **{k.lower(): v for k, v in dict(cfg.MANO).items()}).to(device)
    chunks = []
    total_windows = hands * len(windows(count))
    original_mano = getattr(fitting, 'run_mano', None)
    try:
        fitting.Arguments = licensed_arguments
        # Keep the same left-hand convention as our root/smooth adapter.
        fitting.run_mano = physical_run_mano
        for side in range(hands):
            for start, stop in windows(count):
                indices = padded_indices(start, stop)
                work = Path(output) / 'hmp_windows' / f'{side}_{start:06d}'
                smooth = work / 'smooth_fit'
                smooth.mkdir(parents=True, exist_ok=True)
                selection = {}
                for key, value in base.items():
                    if key != 'betas' and value.ndim >= 2 and value.shape[:2] == (hands, count):
                        selection[key] = value[side:side + 1, indices]
                    elif key == 'betas':
                        selection[key] = value[side:side + 1]
                    else:
                        selection[key] = value.copy()
                np.savez(smooth / 'wilor_000000_world_results.npz', **selection)
                window_obs = {}
                for key, value in obs.items():
                    if torch.is_tensor(value) and value.ndim >= 2 and tuple(value.shape[:2]) == (hands, count):
                        window_obs[key] = value[side:side + 1, indices].clone()
                    elif torch.is_tensor(value) and value.ndim and value.shape[0] == hands:
                        window_obs[key] = value[side:side + 1].clone()
                    else:
                        window_obs[key] = value
                # Padded repetitions carry no image-observation weight.
                if stop - start < WINDOW:
                    window_obs['joints2d'][:, stop - start:, ..., 2] = 0
                    window_obs['vis_mask'][:, stop - start:] = 0
                local = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
                local.HMP.config = options['prior_config']
                local.HMP.vid_path = str(images.resolve())
                local.data.seq = f'wilor_{side}_{start:06d}'
                # A failed rerun must not accept a result from a previous attempt.
                for previous in (work / 'prior').glob('*_world_results.npz'):
                    previous.unlink()
                upstream['run_prior'](local, SimpleNamespace(seq_name=local.data.seq), str(work), device,
                                      ['smooth_fit'], window_obs, model, local, local.data, str(work / 'prior'))
                path = select_result(work, True)
                with np.load(path, allow_pickle=False) as data:
                    chunks.append((side, start, stop, {k: np.asarray(data[k]) for k in data.files}))
                report('HMP 窗口优化', len(chunks), total_windows)
        merged = merge_windows(base, chunks)
        target = Path(output) / 'prior/wilor_000000_world_results.npz'
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix('.tmp')
        with temp.open('wb') as handle:
            np.savez_compressed(handle, **merged)
        temp.replace(target)
        return target
    finally:
        fitting.Arguments = original_arguments
        if original_mano is None:
            del fitting.run_mano
        else:
            fitting.run_mano = original_mano
