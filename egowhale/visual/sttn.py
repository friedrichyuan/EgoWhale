"""STTN weights and inpainting. thirdparty stays unmodified.

The whole frame is painted at half size: STTN resizes every input to its fixed 432x240, so a crop would not
save model time. Windows hold 21 neighbors, 20 apart, so each frame is painted once (the frame two windows share,
twice, averaged); up to 12 references every 10th frame join each window. On EgoDex this matches upstream's +-5 / 5-apart windows at twice the speed.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np

from egowhale.media import CHUNK, SCALE, dilate_masks, read_chunks, resize_frames
from egowhale.paths import THIRDPARTY

_ROOT = THIRDPARTY / "sttn"
_CKPT = _ROOT / "checkpoints" / "sttn.pth"
# The model's input size. Its attention patch sizes divide exactly this feature map.
_WIDTH, _HEIGHT = 432, 240
# Frames moved to the GPU, and through the encoder, at once.
_GPU_CHUNK = 32
_ENCODE_CHUNK = 64


def load_model(device: str = "cuda"):
    """The generator in fp32; paint() runs it under fp16 autocast. It stays on the inpaint actor."""
    import torch

    if not _CKPT.is_file():
        raise FileNotFoundError(_CKPT)
    net = _module().InpaintGenerator(init_weights=False)
    net.load_state_dict(torch.load(_CKPT, map_location="cpu", weights_only=False)["netG"])
    return {"net": net.to(device).eval(), "device": device}


def _module():
    """thirdparty/sttn/model/sttn.py under a private name.

    ProPainter also ships top-level `model` and `core` packages, so neither is put on sys.path. The one import
    the file makes, core.spectral_norm, is served for the duration of the load and then put back.
    """
    name = "_egowhale_sttn"
    if name in sys.modules:
        return sys.modules[name]
    norm = _load("_egowhale_sttn_spectral_norm", _ROOT / "core" / "spectral_norm.py")
    held = {key: sys.modules.get(key) for key in ("core", "core.spectral_norm")}
    package = type(sys)("core")
    package.spectral_norm = norm
    sys.modules.update({"core": package, "core.spectral_norm": norm})
    try:
        module = _load(name, _ROOT / "model" / "sttn.py")
    finally:
        for key, value in held.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    # Upstream's masked_fill is not in place, so the hole mask never reached the attention scores. This is the
    # same math without the -1e9 fill, which overflows in fp16.
    module.Attention.forward = _attention
    return module


def _load(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # A half-run module must not be served by the cache in _module().
        del sys.modules[name]
        raise
    return module


def _attention(_self, query, key, value, _mask):
    import torch.nn.functional as F

    scores = query @ key.transpose(-2, -1) / query.size(-1) ** 0.5
    weights = F.softmax(scores, dim=-1)
    return weights @ value, weights


def prepare_frames(video, start: int, stop: int, masks: np.ndarray, chunk: int = CHUNK) -> dict:
    """Frames [start, stop) at SCALE and their dilated masks, decoded a chunk at a time. Pixels stay uint8."""
    import torch

    height, width = masks.shape[1:]
    size = (int(width * SCALE), int(height * SCALE))
    resized = np.empty((stop - start, size[1], size[0], 3), dtype=np.uint8)
    count = 0
    fps = 30.0
    for fps, block in read_chunks(video, start, stop, chunk):
        resized[count : count + len(block)] = resize_frames(block, size)
        count += len(block)
    if count != len(masks):
        raise ValueError(f"{count} frames vs {len(masks)} masks")
    return {
        "frames": torch.from_numpy(resized).permute(0, 3, 1, 2).unsqueeze(0).contiguous(),
        # Same dilation as ProPainter's input, at the same scale.
        "masks": dilate_masks(masks, size),
        "out_size": size,
        # The painted frames are whole frames; the writer has nothing to paste them into.
        "box": None,
        "fps": float(fps),
    }


def paint_clips(model, clips: list[dict], radius: int = 10, step: int = 20, max_refs: int = 12, fp16: bool = True) -> list[np.ndarray]:
    """Clips one at a time; the windows of one clip already fill the GPU."""
    return [_paint(model, clip, radius, step, max_refs, fp16) for clip in clips]


def windows(length: int, radius: int, step: int, max_refs: int, ref_stride: int = 10) -> list[tuple[list[int], list[int]]]:
    """(neighbors, references) per window. Neighbors are painted; references, the nearest every-10th frames
    outside the window, are only attended to. A last window covers the tail when windows do not overlap."""
    centers = list(range(0, length, step))
    if centers[-1] + radius < length - 1:
        centers.append(length - 1 - radius)
    out = []
    for center in centers:
        neighbors = list(range(max(0, center - radius), min(length, center + radius + 1)))
        refs = [index for index in range(0, length, ref_stride) if index not in neighbors]
        refs = sorted(sorted(refs, key=lambda index: abs(index - center))[:max_refs])
        out.append((neighbors, refs))
    return out


def _paint(model, clip: dict, radius: int, step: int, max_refs: int, fp16: bool) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    net = model["net"]
    device = model["device"]
    frames, masks = clip["frames"][0], clip["masks"][0]
    length = frames.shape[0]
    width, height = clip["out_size"]
    autocast = torch.autocast("cuda", dtype=torch.float16, enabled=fp16)
    with torch.no_grad():
        # Only the 432x240 copies stay on the GPU; full frames go over a chunk at a time, there and back.
        small, holes = [], []
        for begin in range(0, length, _GPU_CHUNK):
            image = frames[begin : begin + _GPU_CHUNK].to(device, non_blocking=True).float().div_(255)
            hole = masks[begin : begin + _GPU_CHUNK].to(device, non_blocking=True).float()
            small.append(F.interpolate(image, size=(_HEIGHT, _WIDTH), mode="bilinear", antialias=True, align_corners=False))
            # Area keeps thin mask edges that nearest would drop; any coverage counts as a hole.
            holes.append((F.interpolate(hole, size=(_HEIGHT, _WIDTH), mode="area") > 0).float())
        small, holes = torch.cat(small), torch.cat(holes)
        with autocast:
            feats = torch.cat([
                net.encoder((small[begin : begin + _ENCODE_CHUNK] * 2 - 1) * (1 - holes[begin : begin + _ENCODE_CHUNK]))
                for begin in range(0, length, _ENCODE_CHUNK)
            ])
        del small
        feat_holes = F.interpolate(holes, scale_factor=0.25)
        channels = feats.shape[1]
        total = torch.zeros(length, 3, _HEIGHT, _WIDTH, device=device)
        count = torch.zeros(length, device=device)
        for neighbors, refs in windows(length, radius, step, max_refs):
            selected = neighbors + refs
            with autocast:
                encoded = net.transformer({"x": feats[selected], "m": feat_holes[selected], "b": 1, "c": channels})["x"]
                decoded = (torch.tanh(net.decoder(encoded[: len(neighbors)])).float() + 1) / 2
            index = torch.tensor(neighbors, device=device)
            total.index_add_(0, index, decoded)
            count.index_add_(0, index, torch.ones(len(neighbors), device=device))
        if not bool((count > 0).all()):
            raise RuntimeError("STTN left a frame unpainted")
        predicted = total / count.view(-1, 1, 1, 1)
        del total, feats
        out = np.empty((length, height, width, 3), dtype=np.uint8)
        for begin in range(0, length, _GPU_CHUNK):
            image = frames[begin : begin + _GPU_CHUNK].to(device, non_blocking=True).float().div_(255)
            hole = masks[begin : begin + _GPU_CHUNK].to(device, non_blocking=True).float()
            up = F.interpolate(predicted[begin : begin + _GPU_CHUNK], size=(height, width), mode="bicubic", align_corners=False)
            mixed = (up * hole + image * (1 - hole)).clamp(0, 1)
            out[begin : begin + len(mixed)] = (mixed * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return out
