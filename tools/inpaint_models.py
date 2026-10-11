"""ProPainter, LaMa, and STTN behind one call: paint(clip) -> uint8 frames at the clip's out_size.

`clip` is what egowhale.visual.paint.prepare_crop returns, so every model sees the same crop, scale, and dilated mask.
Timed sections cover host-to-device, the model, and device-to-host; the CPU crop and the paste are left out.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _to_out(frames: torch.Tensor, out_size: tuple[int, int]) -> np.ndarray:
    """(T,3,H,W) float in [0,1] on GPU -> (T,h,w,3) uint8 at out_size, bicubic like ProPainter's final resize."""
    width, height = out_size
    if frames.shape[-2:] != (height, width):
        frames = F.interpolate(frames, size=(height, width), mode="bicubic", align_corners=False)
    return (frames.clamp(0, 1) * 255 + 0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def _inputs(clip: dict, device) -> tuple[torch.Tensor, torch.Tensor]:
    frames = clip["frames"][0].to(device, non_blocking=True).float().div_(255)
    masks = clip["masks"][0].to(device, non_blocking=True).float()
    return frames, masks


class _Placeholder:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(__import__("pickle").Unpickler):
    """The Lightning checkpoint also pickles its callbacks; only the state dict is needed."""

    def find_class(self, module, name):
        if module.split(".")[0] in ("pytorch_lightning", "lightning_fabric", "lightning"):
            return _Placeholder
        return super().find_class(module, name)


_lenient = types.ModuleType("_lenient")
_lenient.Unpickler = _Unpickler
_lenient.load = lambda file, **kwargs: _Unpickler(file, **kwargs).load()
_lenient.__name__ = "pickle"


class ProPainter:
    name = "propainter"

    def __init__(self, fp16: bool = True, raft_iter: int = 20):
        from egowhale.visual.paint import load_models

        self.models = load_models(str(ROOT / "thirdparty" / "propainter" / "weights"))
        self.fp16 = fp16
        self.raft_iter = raft_iter

    def paint(self, clip: dict) -> np.ndarray:
        from egowhale.visual.paint import paint_clips

        return paint_clips(self.models, [clip], fp16=self.fp16, raft_iter=self.raft_iter)[0]


class LaMa:
    """big-lama generator, frame by frame. Built straight from the FFC module; training deps are stubbed."""

    name = "lama"

    def __init__(self, batch: int = 8, dtype: torch.dtype = torch.float32, device="cuda"):
        root = ROOT / "thirdparty" / "lama"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        # ffc.py imports kornia (spatial transform, off in big-lama) and utils.py imports pytorch_lightning.
        for name in ("kornia", "kornia.geometry", "kornia.geometry.transform"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["kornia.geometry.transform"].rotate = None
        if "pytorch_lightning" not in sys.modules:
            stub = types.ModuleType("pytorch_lightning")
            stub.seed_everything = lambda *args, **kwargs: None
            sys.modules["pytorch_lightning"] = stub
        from omegaconf import OmegaConf
        from saicinpainting.training.modules import ffc
        from saicinpainting.training.modules.ffc import FFCResNetGenerator

        # cuFFT has no half precision for these sizes, so the spectral branch stays fp32 under autocast.
        if not hasattr(ffc.FourierUnit, "_fp32"):
            spectral = ffc.FourierUnit.forward

            def forward(unit, x):
                with torch.autocast("cuda", enabled=False):
                    return spectral(unit, x.float())

            ffc.FourierUnit.forward = forward
            ffc.FourierUnit._fp32 = True

        config = OmegaConf.load(root / "weights" / "big-lama" / "config.yaml")
        kwargs = OmegaConf.to_container(config.generator, resolve=True)
        kwargs.pop("kind")
        net = FFCResNetGenerator(**kwargs)
        state = torch.load(root / "weights" / "big-lama" / "models" / "best.ckpt", map_location="cpu", weights_only=False, pickle_module=_lenient)
        state = {key[len("generator.") :]: value for key, value in state["state_dict"].items() if key.startswith("generator.")}
        net.load_state_dict(state)
        self.net = net.to(device).eval()
        self.batch = batch
        self.dtype = dtype
        self.device = device

    @torch.no_grad()
    def paint(self, clip: dict) -> np.ndarray:
        # Batches move to the GPU one at a time, so memory stays flat with clip length.
        height, width = clip["frames"].shape[-2:]
        # big-lama pads to a multiple of 8; prepare_clip already gives one, this keeps any size safe.
        pad_h, pad_w = -height % 8, -width % 8
        out = []
        for start in range(0, clip["frames"].shape[1], self.batch):
            image = clip["frames"][0, start : start + self.batch].to(self.device, non_blocking=True).float().div_(255)
            mask = clip["masks"][0, start : start + self.batch].to(self.device, non_blocking=True).float()
            x = torch.cat([image * (1 - mask), mask], dim=1)
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            with torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32):
                predicted = self.net(x)
            predicted = predicted[..., :height, :width].float()
            out.append(_to_out(predicted * mask + image * (1 - mask), clip["out_size"]))
        return np.concatenate(out)


class STTN:
    """STTN at its fixed 432x240. The prediction is upsampled and pasted under the full-resolution mask."""

    name = "sttn"
    width, height = 432, 240
    ref_length = 10

    def __init__(
        self,
        batch: int = 4,
        dtype: torch.dtype = torch.float16,
        device="cuda",
        max_refs: int | None = 12,
        radius: int = 5,
        step: int = 5,
        channels_last: bool = False,
    ):
        root = ROOT / "thirdparty" / "sttn"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import model.sttn as sttn
        from model.sttn import InpaintGenerator

        # Upstream's masked_fill is not in place, so the mask never reached the scores. Same math, minus the
        # -1e9 constant that overflows in fp16.
        def attention(_self, query, key, value, _mask):
            scores = torch.matmul(query, key.transpose(-2, -1)) / (query.size(-1) ** 0.5)
            weights = F.softmax(scores, dim=-1)
            return torch.matmul(weights, value), weights

        sttn.Attention.forward = attention

        net = InpaintGenerator()
        net.load_state_dict(torch.load(root / "checkpoints" / "sttn.pth", map_location="cpu", weights_only=False)["netG"])
        self.net = net.to(device).eval()
        self.batch = batch
        self.dtype = dtype
        self.device = device
        # Upstream attends to every 10th frame of the whole clip, so memory grows with length and a 471-frame clip
        # does not fit in 16 GB. Keeping the nearest references bounds it, like ProPainter's ref_num.
        self.max_refs = max_refs
        # Upstream: windows of center +-5, centers 5 apart, so each frame is painted by two windows and averaged.
        self.radius = radius
        self.step = step
        self.channels_last = channels_last
        if channels_last:
            self.net = self.net.to(memory_format=torch.channels_last)

    def _windows(self, length: int) -> list[tuple[list[int], list[int]]]:
        windows = []
        centers = list(range(0, length, self.step))
        # When windows do not overlap, the last one can stop short of the end; one more window covers the tail.
        if centers[-1] + self.radius < length - 1:
            centers.append(length - 1 - self.radius)
        for center in centers:
            neighbors = list(range(max(0, center - self.radius), min(length, center + self.radius + 1)))
            refs = [index for index in range(0, length, self.ref_length) if index not in neighbors]
            if self.max_refs is not None and len(refs) > self.max_refs:
                refs = sorted(sorted(refs, key=lambda index: abs(index - center))[: self.max_refs])
            windows.append((neighbors, refs))
        return windows

    @torch.no_grad()
    def paint(self, clip: dict) -> np.ndarray:
        # Only 432x240 copies stay on the GPU; full-size frames move over a batch at a time, in and out,
        # so memory does not grow with clip length.
        length = clip["frames"].shape[1]
        size = tuple(clip["frames"].shape[-2:])
        step = 32
        small, small_mask = [], []
        for i in range(0, length, step):
            frames = clip["frames"][0, i : i + step].to(self.device, non_blocking=True).float().div_(255)
            masks = clip["masks"][0, i : i + step].to(self.device, non_blocking=True).float()
            small.append(F.interpolate(frames, size=(self.height, self.width), mode="bilinear", antialias=True, align_corners=False))
            # Area keeps thin mask edges that nearest would drop; any coverage counts as a hole.
            small_mask.append((F.interpolate(masks, size=(self.height, self.width), mode="area") > 0).float())
        small, small_mask = torch.cat(small), torch.cat(small_mask)
        net = self.net
        autocast = torch.autocast("cuda", dtype=self.dtype, enabled=self.dtype != torch.float32)
        layout = torch.channels_last if self.channels_last else torch.contiguous_format
        with autocast:
            feats = torch.cat([
                net.encoder(((small[i : i + 64] * 2 - 1) * (1 - small_mask[i : i + 64])).contiguous(memory_format=layout))
                for i in range(0, length, 64)
            ])
        feat_mask = F.interpolate(small_mask, scale_factor=0.25)
        channels = feats.shape[1]
        del small

        # Windows with the same frame count share one transformer call; upstream runs them one by one.
        groups: dict[int, list] = {}
        for window in self._windows(length):
            groups.setdefault(len(window[0]) + len(window[1]), []).append(window)
        total = torch.zeros(length, 3, self.height, self.width, device=self.device)
        count = torch.zeros(length, device=self.device)
        for windows in groups.values():
            for start in range(0, len(windows), self.batch):
                part = windows[start : start + self.batch]
                selected = [neighbors + refs for neighbors, refs in part]
                flat = [index for indexes in selected for index in indexes]
                with autocast:
                    enc = net.transformer({"x": feats[flat], "m": feat_mask[flat], "b": len(part), "c": channels})["x"]
                    enc = enc.view(len(part), -1, *enc.shape[1:])
                    keep = torch.cat([enc[b, : len(neighbors)] for b, (neighbors, _refs) in enumerate(part)])
                    decoded = (torch.tanh(net.decoder(keep)).float() + 1) / 2
                # Upstream blends overlapping windows 0.5/0.5 in window order; an even mean over windows is order-free.
                index = torch.tensor([i for neighbors, _refs in part for i in neighbors], device=self.device)
                total.index_add_(0, index, decoded)
                count.index_add_(0, index, torch.ones(len(index), device=self.device))
        if not bool((count > 0).all()):
            raise RuntimeError("STTN left a frame unpainted")
        predicted = total / count.view(-1, 1, 1, 1)
        del total, feats
        out = []
        for i in range(0, length, step):
            frames = clip["frames"][0, i : i + step].to(self.device, non_blocking=True).float().div_(255)
            masks = clip["masks"][0, i : i + step].to(self.device, non_blocking=True).float()
            up = F.interpolate(predicted[i : i + step], size=size, mode="bicubic", align_corners=False)
            out.append(_to_out(up * masks + frames * (1 - masks), clip["out_size"]))
        return np.concatenate(out)
