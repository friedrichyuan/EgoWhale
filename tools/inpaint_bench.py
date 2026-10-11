"""Inpainter comparison on EgoDex: GPU throughput, synthetic-hole quality, and side-by-side videos.

Each method runs in its own process (ProPainter and STTN both ship a top-level `model` package):

    python tools/inpaint_bench.py run  --method lama   # per-episode timing, metrics, and output videos
    python tools/inpaint_bench.py sweep --method lama  # batch / precision sweep for the best fps
    python tools/inpaint_bench.py grid                 # comparison videos and the summary table

Real masks come from SAM 3 (the segment stage). Quality has no ground truth there, so it is also measured with
synthetic holes: another episode's person mask is laid over this episode, the union is painted, and only pixels in
the borrowed mask that are well away from the real person are scored against the original frame.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "thirdparty" / "pylibs"))

from egowhale.media import load_masks, read_rgb  # noqa: E402
from egowhale.media import resize_frames as _resize_all  # noqa: E402
from egowhale.visual.paint import crop_box, prepare_clip  # noqa: E402

# "mask": the pipeline's crop_box. "full": the whole frame. "aspect": crop_box grown to STTN's 432:240 so the
# model's fixed resize does not stretch it.
BOX = "mask"


def box_for(masks: np.ndarray, mode: str) -> tuple[int, int, int, int]:
    height, width = masks.shape[1:]
    if mode == "full":
        return 0, height, 0, width
    y0, y1, x0, x1 = crop_box(masks)
    if mode == "mask":
        return y0, y1, x0, x1
    if mode != "aspect":
        raise ValueError(mode)
    target = 432 / 240
    box_h, box_w = y1 - y0, x1 - x0
    if box_w / box_h > target:
        box_h = min(height, int(np.ceil(box_w / target / 16)) * 16)
    else:
        box_w = min(width, int(np.ceil(box_h * target / 16)) * 16)
    # Grow around the original center, then shift back inside the frame.
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    y0 = int(np.clip(cy - box_h // 2, 0, height - box_h)) // 16 * 16
    x0 = int(np.clip(cx - box_w // 2, 0, width - box_w)) // 16 * 16
    return y0, min(height, y0 + box_h), x0, min(width, x0 + box_w)


def prepare_crop(frames: np.ndarray, masks: np.ndarray, resize_ratio: float = 0.5) -> dict:
    """The benchmark's in-memory crop: the pipeline's crop_box and prepare_clip, plus the resized background
    the pipeline now re-decodes at write time. Same box and pixels as the stage."""
    y0, y1, x0, x1 = box_for(masks, BOX)
    clip = prepare_clip(np.ascontiguousarray(frames[:, y0:y1, x0:x1]), masks[:, y0:y1, x0:x1], resize_ratio)
    size = (int(frames.shape[2] * resize_ratio), int(frames.shape[1] * resize_ratio))
    clip["background"] = _resize_all(frames, size)
    clip["box"] = (int(y0 * resize_ratio), int(x0 * resize_ratio))
    return clip


def paste(background: np.ndarray, painted: np.ndarray, box: tuple[int, int]) -> np.ndarray:
    out = np.array(background)
    top, left = box
    out[:, top : top + painted.shape[1], left : left + painted.shape[2]] = painted
    return out

BENCH = ROOT / "outputs" / "inpaint_bench"
DATA = ROOT / "datasets" / "egodex" / "test"
METHODS = ("propainter", "lama", "sttn")
LABELS = {"propainter": "ProPainter", "lama": "LaMa (big-lama)", "sttn": "STTN"}
# The scored region stays this far (full-resolution pixels) from the real person mask.
GUARD = 25


def episodes() -> list[str]:
    return [line.strip() for line in (BENCH / "episodes.txt").read_text().splitlines() if line.strip()]


def load(episode: str) -> tuple[np.ndarray, np.ndarray, float]:
    task, index = episode.split("/")
    frames, fps = read_rgb(DATA / task / f"{index}.mp4")
    masks = load_masks(BENCH / "eps" / f"{task}_{index}" / "visual" / "masks.npz")
    if len(frames) != len(masks):
        raise ValueError(f"{episode}: {len(frames)} frames vs {len(masks)} masks")
    return frames, masks, fps


def synthetic(masks: np.ndarray, donor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Painted mask (real | borrowed) and the scored region (borrowed, minus the guarded real person)."""
    borrowed = donor[np.arange(len(masks)) % len(donor)]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GUARD + 1, 2 * GUARD + 1))
    guard = np.stack([cv2.dilate(mask.astype(np.uint8), kernel) for mask in masks]).astype(bool)
    return masks | borrowed, borrowed & ~guard


def build(name: str, **kwargs):
    import torch
    from inpaint_models import STTN, LaMa, ProPainter

    defaults = {
        "propainter": (ProPainter, {}),
        "lama": (LaMa, {"batch": 4, "dtype": torch.float16}),
        "sttn": (STTN, {"batch": 1, "dtype": torch.float16}),
    }
    cls, base = defaults[name]
    return cls(**{**base, **kwargs})


def timed(model, clip, chunk: int | None = None, overlap: int = 0) -> tuple[np.ndarray, float, float]:
    """Paint in temporal chunks of `chunk` frames (all at once when None). Each chunk is painted with
    `overlap` extra frames of context on both sides, which are then dropped."""
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    length = clip["frames"].shape[1]
    step = chunk or length
    parts = []
    for a in range(0, length, step):
        b = min(a + step, length)
        left, right = max(0, a - overlap), min(length, b + overlap)
        parts.append(model.paint(_sub(clip, left, right))[a - left : a - left + b - a])
    painted = np.concatenate(parts)
    torch.cuda.synchronize()
    return painted, time.perf_counter() - started, torch.cuda.max_memory_allocated() / 2**30


def _sub(clip: dict, start: int, stop: int) -> dict:
    if start == 0 and stop == clip["frames"].shape[1]:
        return clip
    part = dict(clip)
    part["frames"] = clip["frames"][:, start:stop]
    part["masks"] = clip["masks"][:, start:stop]
    part["key"] = (*clip["key"][:2], stop - start)
    return part


def fitted(model, clip, chunk: int | None) -> tuple[np.ndarray, float, float, int | None]:
    """Whole clip first; on OOM halve the chunk until it fits. Returns the chunk that worked."""
    import torch

    length = clip["frames"].shape[1]
    while True:
        try:
            return (*timed(model, clip, chunk), chunk)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            chunk = (chunk or length) // 2
            if chunk < 20:
                raise
            print(f"  OOM, retrying in chunks of {chunk}", flush=True)


def write_h264(path: Path, frames: np.ndarray, fps: float, crf: int = 16) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=2, ffmpeg_params=["-crf", str(crf), "-pix_fmt", "yuv420p"]) as writer:
        for frame in frames:
            writer.append_data(frame)


def read_video(path: Path) -> np.ndarray:
    # imageio's reader does not stop at the end of these files, so cv2 decodes them.
    return read_rgb(path)[0]


class Scorer:
    """PSNR, SSIM, LPIPS, and temporal error inside the scored region, at the painted (half) resolution."""

    def __init__(self):
        import lpips
        import torch

        self.torch = torch
        self.lpips = lpips.LPIPS(net="alex", verbose=False).cuda().eval()

    def __call__(self, out: np.ndarray, truth: np.ndarray, region: np.ndarray) -> dict:
        torch = self.torch
        sq, ssim_sum, lpips_vals, temporal, pixels, tpixels = 0.0, 0.0, [], 0.0, 0, 0
        previous = None
        pair_error, pair_pixels = [], []
        window = _gaussian(11, 1.5).cuda()
        for start in range(0, len(out), 32):
            o = torch.from_numpy(out[start : start + 32]).cuda().permute(0, 3, 1, 2).float()
            g = torch.from_numpy(truth[start : start + 32]).cuda().permute(0, 3, 1, 2).float()
            r = torch.from_numpy(region[start : start + 32]).cuda().unsqueeze(1).float()
            sq += float((((o - g) ** 2) * r).sum()) / 3
            pixels += float(r.sum())
            ssim_sum += float((_ssim(o, g, window) * r).sum())
            # Temporal error: frame-to-frame change of the output against that of the original.
            if previous is not None:
                po, pg, pr = (torch.cat([p, c]) for p, c in zip(previous, (o, g, r)))
            else:
                po, pg, pr = o, g, r
            both = pr[1:] * pr[:-1]
            error = ((po[1:] - po[:-1]) - (pg[1:] - pg[:-1])).abs().mean(1, keepdim=True).mul(both)
            temporal += float(error.sum())
            tpixels += float(both.sum())
            pair_error.extend(error.flatten(1).sum(1).tolist())
            pair_pixels.extend(both.flatten(1).sum(1).tolist())
            previous = (o[-1:], g[-1:], r[-1:])
            # LPIPS sees only the scored change: outside the region both images are the original.
            ys, xs = torch.where(r.amax(0)[0] > 0)
            if len(ys):
                y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
                mixed = (o * r + g * (1 - r))[..., y0:y1, x0:x1] / 127.5 - 1
                lpips_vals.append(float(self.lpips(mixed, g[..., y0:y1, x0:x1] / 127.5 - 1).sum()))
        mse = sq / max(pixels, 1)
        return {
            "psnr": 10 * np.log10(255**2 / max(mse, 1e-8)),
            "ssim": ssim_sum / max(pixels, 1),
            "lpips": sum(lpips_vals) / len(out),
            "temporal": temporal / max(tpixels, 1),
            "scored_pixels": int(pixels),
            # Pair t is frames (t, t+1); kept so chunk seams can be scored apart from the rest.
            "pair_error": pair_error,
            "pair_pixels": pair_pixels,
        }


def _gaussian(size: int, sigma: float):
    import torch

    x = torch.arange(size).float() - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = (g / g.sum()).unsqueeze(1)
    return (g @ g.T).expand(3, 1, size, size).contiguous()


def _ssim(x, y, window):
    """Per-pixel SSIM averaged over channels, 0-255 inputs."""
    import torch.nn.functional as F

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    pad = window.shape[-1] // 2
    mu_x = F.conv2d(x, window, padding=pad, groups=3)
    mu_y = F.conv2d(y, window, padding=pad, groups=3)
    sxx = F.conv2d(x * x, window, padding=pad, groups=3) - mu_x**2
    syy = F.conv2d(y * y, window, padding=pad, groups=3) - mu_y**2
    sxy = F.conv2d(x * y, window, padding=pad, groups=3) - mu_x * mu_y
    value = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return value.mean(1, keepdim=True)


def half(masks: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.stack([cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) for mask in masks]).astype(bool)


def run(method: str, tag: str | None = None, chunk: int | None = None, overlap: int = 0, skip_real: bool = False, **kwargs) -> None:
    """`tag` names the result folder for a variant; `chunk`/`overlap` fix the temporal slicing instead of
    letting OOM pick it; extra kwargs go to the model."""
    model = build(method, **kwargs)
    scorer = Scorer()
    names = episodes()
    out_dir = BENCH / "results" / (tag or method)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # Warm-up: cuDNN autotuning and lazy init stay out of the first timed episode.
    frames, masks, _fps = load(names[0])
    model.paint(prepare_crop(frames[:40], masks[:40]))
    donor_cache = {}
    fixed = chunk is not None
    for position, episode in enumerate(names):
        frames, masks, fps = load(episode)
        tag = episode.replace("/", "_")
        row = {"episode": episode, "frames": len(frames)}

        if not skip_real:
            started = time.perf_counter()
            clip = prepare_crop(frames, masks)
            row["prep_seconds"] = time.perf_counter() - started
            if fixed:
                painted, seconds, memory = timed(model, clip, chunk, overlap)
            else:
                painted, seconds, memory, chunk = fitted(model, clip, None)
            row.update(chunk=chunk, real_seconds=seconds, real_fps=len(frames) / seconds, real_peak_gb=memory, crop=list(clip["out_size"]))
            write_h264(out_dir / f"{tag}_real.mp4", paste(clip["background"], painted, clip["box"]), fps)
            del clip, painted

        donor_name = names[(position + 1) % len(names)]
        if donor_name not in donor_cache:
            donor_cache = {donor_name: load(donor_name)[1]}
        painted_mask, region = synthetic(masks, donor_cache[donor_name])
        clip = prepare_crop(frames, painted_mask)
        if fixed:
            painted, seconds, memory = timed(model, clip, chunk, overlap)
        else:
            painted, seconds, memory, chunk = fitted(model, clip, None)
        row.update(chunk=chunk, overlap=overlap)
        size = (clip["background"].shape[2], clip["background"].shape[1])
        out = paste(clip["background"], painted, clip["box"])
        scores = scorer(out, clip["background"], half(region, size))
        row.update(syn_seconds=seconds, syn_fps=len(frames) / seconds, syn_peak_gb=memory, donor=donor_name, **{f"syn_{k}": v for k, v in scores.items()})
        write_h264(out_dir / f"{tag}_syn.mp4", out, fps)
        np.savez_compressed(out_dir / f"{tag}_syn_region.npz", region=np.packbits(half(region, size), axis=-1), shape=np.array(half(region, size).shape))
        rows.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if not isinstance(v, list)}), flush=True)
        del clip, painted, out, frames, masks
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=1))


def sweep(method: str, episode: str) -> None:
    """Best single-GPU fps over batch and precision, on one real-mask episode. Data is prepared once."""
    import torch

    frames, masks, _fps = load(episode)
    clip = prepare_crop(frames, masks)
    del frames
    if method == "lama":
        grid = [dict(batch=b, dtype=d) for d in (torch.float32, torch.float16) for b in (1, 4, 8, 16, 32, 48)]
    elif method == "sttn":
        grid = [dict(batch=b, dtype=d) for d in (torch.float32, torch.float16) for b in (1, 2, 4, 8, 16)]
    else:
        grid = [dict(fp16=True), dict(fp16=False), dict(fp16=True, raft_iter=10)]
    rows = []
    model = None
    for config in grid:
        try:
            if method == "propainter" or model is None:
                del model
                torch.cuda.empty_cache()
                model = build(method, **config)
            else:
                for key, value in config.items():
                    setattr(model, key, value)
            model.paint(clip)
            best = min(timed(model, clip)[1] for _ in range(3))
            memory = timed(model, clip)[2]
            row = {"config": {k: str(v).replace("torch.", "") for k, v in config.items()}, "fps": len(masks) / best, "peak_gb": memory}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            row = {"config": {k: str(v).replace("torch.", "") for k, v in config.items()}, "fps": None, "peak_gb": "OOM"}
        rows.append(row)
        print(json.dumps(row), flush=True)
    path = BENCH / "results" / f"sweep_{method}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"episode": episode, "frames": len(masks), "crop": list(clip["out_size"]), "rows": rows}, indent=1))


def scaling(method: str, lengths=(60, 120, 240, 471)) -> None:
    """fps against clip length on the longest episode. STTN's references grow with length."""
    frames, masks, _fps = load("basic_fold/0")
    model = build(method)
    model.paint(prepare_crop(frames[:40], masks[:40]))
    rows = []
    for length in lengths:
        clip = prepare_crop(frames[:length], masks[:length])
        seconds = min(timed(model, clip)[1] for _ in range(2))
        rows.append({"frames": length, "fps": length / seconds})
        print(json.dumps(rows[-1]), flush=True)
    (BENCH / "results" / f"scaling_{method}.json").write_text(json.dumps(rows, indent=1))


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    scale = tile.shape[0] / 540
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, 2)
    cv2.rectangle(tile, (0, 0), (w + int(20 * scale), h + int(20 * scale)), (0, 0, 0), -1)
    cv2.putText(tile, text, (int(10 * scale), h + int(10 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * scale, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def _overlay(frames: np.ndarray, masks: np.ndarray, color=(255, 60, 60)) -> np.ndarray:
    out = frames.copy()
    tint = np.array(color, dtype=np.float32)
    out[masks] = (out[masks] * 0.5 + tint * 0.5).astype(np.uint8)
    for frame, mask in zip(out, masks):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)
    return out


def grid() -> None:
    """2x2 videos per episode: input with mask | ProPainter / LaMa | STTN. Real masks and synthetic holes.

    Frames stream through readers and writers, so a long episode never sits in memory four times over.
    """
    import imageio.v2 as imageio

    names = episodes()
    sweeps = {m: json.loads((BENCH / "results" / f"sweep_{m}.json").read_text()) for m in METHODS if (BENCH / "results" / f"sweep_{m}.json").is_file()}
    metrics = {m: json.loads((BENCH / "results" / m / "metrics.json").read_text()) for m in METHODS}
    videos = BENCH / "videos"
    videos.mkdir(parents=True, exist_ok=True)
    options = dict(codec="libx264", macro_block_size=2)
    reel = imageio.get_writer(videos / "all_real.mp4", fps=30, ffmpeg_params=["-crf", "22", "-pix_fmt", "yuv420p"], **options)
    for position, episode in enumerate(names):
        tag = episode.replace("/", "_")
        task, index = episode.split("/")
        masks = load_masks(BENCH / "eps" / f"{task}_{index}" / "visual" / "masks.npz")
        capture = cv2.VideoCapture(str(DATA / task / f"{index}.mp4"))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
        capture.release()
        stride = 2 if len(masks) > 150 else 1
        for kind in ("real", "syn"):
            if kind == "real":
                hole, scored, first_label = masks, None, "Input + SAM3 mask"
            else:
                with np.load(BENCH / "results" / "lama" / f"{tag}_syn_region.npz") as data:
                    shape = tuple(data["shape"])
                    scored = np.unpackbits(data["region"], axis=-1)[..., : shape[-1]].astype(bool)
                hole, _ = synthetic(masks, load_masks(_mask_path(names[(position + 1) % len(names)])))
                first_label = "Original (grey = hole, green = scored)"
            capture = cv2.VideoCapture(str(DATA / task / f"{index}.mp4"))
            readers = {m: imageio.get_reader(BENCH / "results" / m / f"{tag}_{kind}.mp4") for m in METHODS}
            writer = imageio.get_writer(videos / f"{tag}_{kind}.mp4", fps=fps, ffmpeg_params=["-crf", "20", "-pix_fmt", "yuv420p"], **options)
            for t, outs in enumerate(zip(*(readers[m] for m in METHODS))):
                ok, frame = capture.read()
                if not ok:
                    break
                size = (frame.shape[1] // 2, frame.shape[0] // 2)
                # Older result videos were padded to a multiple of 8 by the writer; bring them back to the true size.
                outs = {m: o if o.shape[:2] == size[::-1] else cv2.resize(o, size, interpolation=cv2.INTER_AREA) for m, o in zip(METHODS, outs)}
                small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), size, interpolation=cv2.INTER_AREA)
                first = _overlay(small[None], half(hole[t : t + 1], size), (90, 90, 90) if scored is not None else (255, 60, 60))
                if scored is not None:
                    first = _overlay(first, scored[t : t + 1], (60, 220, 60))
                row0 = np.concatenate([_label(first[0], first_label), _label(outs["propainter"].copy(), LABELS["propainter"])], 1)
                row1 = np.concatenate([_label(outs["lama"].copy(), LABELS["lama"]), _label(outs["sttn"].copy(), LABELS["sttn"])], 1)
                tile = np.concatenate([row0, row1], 0)
                writer.append_data(tile)
                if kind == "real" and t % stride == 0:
                    reel.append_data(_label(tile, episode) if t == 0 else tile)
            writer.close()
            capture.release()
            for reader in readers.values():
                reader.close()
            print("wrote", videos / f"{tag}_{kind}.mp4", flush=True)
    reel.close()
    summary(metrics, sweeps)


def _mask_path(episode: str) -> Path:
    task, index = episode.split("/")
    return BENCH / "eps" / f"{task}_{index}" / "visual" / "masks.npz"


def summary(metrics: dict, sweeps: dict) -> None:
    lines = ["| Method | fps (real masks, mean) | Peak GB | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Temporal err ↓ |", "|---|---|---|---|---|---|---|"]
    table = {}
    for method in METHODS:
        rows = metrics[method]
        frames = sum(r["frames"] for r in rows)
        fps = frames / sum(r["real_seconds"] for r in rows)
        weight = np.array([r["frames"] for r in rows], dtype=float)
        avg = {k: float(np.average([r[f"syn_{k}"] for r in rows], weights=weight)) for k in ("psnr", "ssim", "lpips", "temporal")}
        peak = max(r["real_peak_gb"] for r in rows)
        table[method] = {"fps": fps, "peak_gb": peak, **avg}
        lines.append(f"| {LABELS[method]} | {fps:.1f} | {peak:.1f} | {avg['psnr']:.2f} | {avg['ssim']:.4f} | {avg['lpips']:.4f} | {avg['temporal']:.2f} |")
    for method, data in sweeps.items():
        best = max((r for r in data["rows"] if r["fps"]), key=lambda r: r["fps"])
        table[method]["sweep_best"] = best
    (BENCH / "summary.md").write_text("\n".join(lines) + "\n")
    (BENCH / "summary.json").write_text(json.dumps(table, indent=1))
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("run", "sweep", "scaling", "grid"))
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--episode", default="assemble_disassemble_legos/0")
    parser.add_argument("--tag")
    parser.add_argument("--dtype", choices=("float16", "float32"))
    parser.add_argument("--max-refs", type=int, help="STTN reference cap; 0 keeps every 10th frame of the clip")
    parser.add_argument("--chunk", type=int)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--radius", type=int, help="STTN neighbors on each side of a window center")
    parser.add_argument("--step", type=int, help="STTN distance between window centers")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--box", choices=("mask", "full", "aspect"), default="mask")
    args = parser.parse_args()
    global BOX
    BOX = args.box
    if args.mode == "run":
        import torch

        kwargs = {}
        if args.dtype:
            kwargs["dtype"] = getattr(torch, args.dtype)
        if args.max_refs is not None:
            kwargs["max_refs"] = args.max_refs or None
        for key in ("radius", "step"):
            if getattr(args, key) is not None:
                kwargs[key] = getattr(args, key)
        if args.channels_last:
            kwargs["channels_last"] = True
        run(args.method, args.tag, args.chunk, args.overlap, args.skip_real, **kwargs)
    elif args.mode == "sweep":
        sweep(args.method, args.episode)
    elif args.mode == "scaling":
        scaling(args.method)
    else:
        grid()


if __name__ == "__main__":
    main()
