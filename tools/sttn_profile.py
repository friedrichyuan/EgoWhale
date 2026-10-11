"""Where STTN's time goes on one episode, stage by stage, plus exact-math speedups (layout, cuDNN autotune).

    python tools/sttn_profile.py [episode] [--radius R --step S --max-refs N --batch B]

CPU stages use the pipeline's own prepare_crop / write_painted, so they are what the inpaint node would pay.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from inpaint_models import STTN, _to_out  # noqa: E402

from egowhale.media import load_masks  # noqa: E402
from egowhale.visual.paint import prepare_crop, write_painted  # noqa: E402

BENCH = ROOT / "outputs" / "inpaint_bench"
DATA = ROOT / "datasets" / "egodex" / "test"


class Clock:
    """CUDA-event timer per named stage; totals in seconds."""

    def __init__(self):
        self.marks: dict[str, list] = {}

    def section(self, name):
        clock = self

        class _Section:
            def __enter__(self):
                self.start = torch.cuda.Event(enable_timing=True)
                self.stop = torch.cuda.Event(enable_timing=True)
                self.start.record()

            def __exit__(self, *_):
                self.stop.record()
                clock.marks.setdefault(name, []).append((self.start, self.stop))

        return _Section()

    def totals(self) -> dict[str, float]:
        torch.cuda.synchronize()
        return {name: sum(a.elapsed_time(b) for a, b in pairs) / 1000 for name, pairs in self.marks.items()}


@torch.no_grad()
def staged(model: STTN, clip: dict, clock: Clock) -> np.ndarray:
    """STTN.paint with each stage under its own timer. Same math as paint()."""
    length = clip["frames"].shape[1]
    size = tuple(clip["frames"].shape[-2:])
    step = 32
    net = model.net
    autocast = torch.autocast("cuda", dtype=model.dtype, enabled=model.dtype != torch.float32)
    layout = torch.channels_last if model.channels_last else torch.contiguous_format
    with clock.section("h2d + downsample"):
        small, small_mask = [], []
        for i in range(0, length, step):
            frames = clip["frames"][0, i : i + step].to(model.device, non_blocking=True).float().div_(255)
            masks = clip["masks"][0, i : i + step].to(model.device, non_blocking=True).float()
            small.append(F.interpolate(frames, size=(model.height, model.width), mode="bilinear", antialias=True, align_corners=False))
            small_mask.append((F.interpolate(masks, size=(model.height, model.width), mode="area") > 0).float())
        small, small_mask = torch.cat(small), torch.cat(small_mask)
    with clock.section("encoder"), autocast:
        feats = torch.cat([
            net.encoder(((small[i : i + 64] * 2 - 1) * (1 - small_mask[i : i + 64])).contiguous(memory_format=layout))
            for i in range(0, length, 64)
        ])
    feat_mask = F.interpolate(small_mask, scale_factor=0.25)
    channels = feats.shape[1]
    groups: dict[int, list] = {}
    windows = model._windows(length)
    for window in windows:
        groups.setdefault(len(window[0]) + len(window[1]), []).append(window)
    total = torch.zeros(length, 3, model.height, model.width, device=model.device)
    count = torch.zeros(length, device=model.device)
    transformer_frames = 0
    for group in groups.values():
        for start in range(0, len(group), model.batch):
            part = group[start : start + model.batch]
            flat = [index for neighbors, refs in part for index in neighbors + refs]
            transformer_frames += len(flat)
            with clock.section("transformer"), autocast:
                enc = net.transformer({"x": feats[flat], "m": feat_mask[flat], "b": len(part), "c": channels})["x"]
            with clock.section("decoder"), autocast:
                enc = enc.view(len(part), -1, *enc.shape[1:])
                keep = torch.cat([enc[b, : len(neighbors)] for b, (neighbors, _refs) in enumerate(part)])
                decoded = (torch.tanh(net.decoder(keep)).float() + 1) / 2
            index = torch.tensor([i for neighbors, _refs in part for i in neighbors], device=model.device)
            total.index_add_(0, index, decoded)
            count.index_add_(0, index, torch.ones(len(index), device=model.device))
    predicted = total / count.view(-1, 1, 1, 1)
    out = []
    with clock.section("upsample + paste + d2h"):
        for i in range(0, length, step):
            frames = clip["frames"][0, i : i + step].to(model.device, non_blocking=True).float().div_(255)
            masks = clip["masks"][0, i : i + step].to(model.device, non_blocking=True).float()
            up = F.interpolate(predicted[i : i + step], size=size, mode="bicubic", align_corners=False)
            out.append(_to_out(up * masks + frames * (1 - masks), clip["out_size"]))
    clock.windows = len(windows)
    clock.transformer_frames = transformer_frames
    return np.concatenate(out)


def gpu_fps(model: STTN, clip: dict, repeats: int = 3) -> float:
    model.paint(clip)
    best = float("inf")
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        model.paint(clip)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - started)
    return clip["frames"].shape[1] / best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", nargs="?", default="basic_fold/0")
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--max-refs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    task, index = args.episode.split("/")
    video = DATA / task / f"{index}.mp4"
    masks = load_masks(BENCH / "eps" / f"{task}_{index}" / "visual" / "masks.npz")
    length = len(masks)

    started = time.perf_counter()
    clip = prepare_crop(video, 0, length, masks)
    cpu_prepare = time.perf_counter() - started
    print(f"{args.episode}: {length} frames, crop out {clip['out_size']}, process {tuple(clip['frames'].shape[-2:][::-1])}")

    knobs = dict(radius=args.radius, step=args.step, max_refs=args.max_refs or None, batch=args.batch)
    model = STTN(dtype=torch.float16, **knobs)
    staged(model, clip, Clock())  # warm-up
    clock = Clock()
    torch.cuda.synchronize()
    started = time.perf_counter()
    painted = staged(model, clip, clock)
    torch.cuda.synchronize()
    gpu_wall = time.perf_counter() - started
    totals = clock.totals()

    with tempfile.TemporaryDirectory() as tmp:
        started = time.perf_counter()
        write_painted(Path(tmp) / "inpaint.mp4", video, 0, length, painted, clip["box"])
        cpu_write = time.perf_counter() - started

    print(f"windows {clock.windows}, transformer frames {clock.transformer_frames} ({clock.transformer_frames / length:.2f} per output frame)")
    print(f"{'stage':<26}{'seconds':>9}{'share':>8}{'fps if alone':>14}")
    for name, seconds in totals.items():
        print(f"{name:<26}{seconds:>9.2f}{seconds / gpu_wall:>8.0%}{length / seconds:>14.1f}")
    print(f"{'GPU wall (sum)':<26}{gpu_wall:>9.2f}{'':>8}{length / gpu_wall:>14.1f}")
    print(f"{'CPU prepare_crop':<26}{cpu_prepare:>9.2f}{'':>8}{length / cpu_prepare:>14.1f}")
    print(f"{'CPU write_painted':<26}{cpu_write:>9.2f}{'':>8}{length / cpu_write:>14.1f}")

    print("\nexact-math variants (GPU fps, best of 3):")
    for label, setup in [
        ("baseline", lambda: None),
        ("cudnn.benchmark", lambda: setattr(torch.backends.cudnn, "benchmark", True)),
    ]:
        setup()
        for layout in (False, True):
            variant = STTN(dtype=torch.float16, channels_last=layout, **knobs)
            print(f"  {label:<16} channels_last={layout!s:<5} {gpu_fps(variant, clip):6.1f} fps")
            del variant
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
