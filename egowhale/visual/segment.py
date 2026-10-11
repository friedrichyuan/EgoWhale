"""SAM 3 keyframes plus Cutie propagation. Both weights stay on the segment actor."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from egowhale import thirdparty
from egowhale.media import read_rgb, save_masks
from egowhale.paths import MASKS, THIRDPARTY
from egowhale.step import Result, Step, failed_items, short_error

_CKPT = THIRDPARTY / "sam3" / "weights" / "sam3" / "sam3.pt"
_BPE = THIRDPARTY / "sam3" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
_CUTIE = THIRDPARTY / "cutie" / "weights" / "cutie-base-mega.pth"
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Every _STRIDE-th frame is a SAM 3 keyframe; Cutie carries the mask between them at _SHORT pixels on the short side.
_STRIDE = 10
_SHORT = 480


def load_tracker():
    """Official SAM 3 image model and Cutie. EfficientSAM3 must not share this process."""
    if not _CKPT.is_file():
        raise FileNotFoundError(_CKPT)
    thirdparty.use_sam3()
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=str(_CKPT),
        bpe_path=str(_BPE),
        load_from_HF=False,
        compile=False,
        device="cuda",
    )
    return Sam3Processor(model, confidence_threshold=0.5), _load_cutie()


def _load_cutie():
    import torch
    from omegaconf import OmegaConf

    thirdparty.use("cutie")
    import tracker.model.utils.resnet as resnet
    from tracker.config import CONFIG
    from tracker.inference.inference_core import InferenceCore
    from tracker.model.cutie import CUTIE

    # Cutie's backbones would download ImageNet weights that its own checkpoint then replaces.
    resnet.resnet18 = lambda pretrained=True, extra_dim=0: resnet.ResNet(resnet.BasicBlock, [2, 2, 2, 2], extra_dim)
    resnet.resnet50 = lambda pretrained=True, extra_dim=0: resnet.ResNet(resnet.Bottleneck, [3, 4, 6, 3], extra_dim)
    cfg = OmegaConf.create(CONFIG)
    OmegaConf.resolve(cfg)
    cfg.max_internal_size = -1
    network = CUTIE(cfg).cuda().eval()
    network.load_weights(torch.load(_CUTIE, map_location="cuda", weights_only=False))
    return InferenceCore(network, cfg)


class Segment(Step):
    name = "segment"
    makes = (MASKS,)
    gpus = 1
    per_piece = True

    def __init__(self) -> None:
        super().__init__()
        self.processor = None
        self.cutie = None

    def load(self) -> None:
        if self.processor is None:
            self.processor, self.cutie = load_tracker()

    def release(self) -> None:
        self.processor = self.cutie = None

    def run_piece(self, src: Path, dst: Path, index: int) -> None:
        item = self.prepare(src, *self.piece(dst, index))
        self.load()
        save_masks(Path(item["dst"]), segment_video(self.processor, self.cutie, item["frames"]))

    def prepare(self, src: Path, where: Path, start: int, stop: int) -> dict:
        """The piece's frames at full size, for a CPU reader."""
        video = Path(src).with_suffix(".mp4")
        if not video.is_file():
            raise FileNotFoundError(video)
        frames, _fps = read_rgb(video, start, stop)
        return {"frames": frames, "dst": str(Path(where) / MASKS)}

    def consume(self, payload: dict, budget: int | None = None) -> dict:
        """Segment prefetched pieces one at a time. Masks go back for a CPU writer; `budget` is unused."""
        import ray

        items = payload["items"]
        results = failed_items(items)
        saves = []
        self.load()
        with self.lock:
            for index, item in enumerate(items):
                if results[index] is not None:
                    continue
                started = time.perf_counter()
                try:
                    masks = segment_video(self.processor, self.cutie, item["frames"])
                except Exception as exc:
                    results[index] = Result(False, 0.0, short_error(exc))
                    continue
                results[index] = Result(True, time.perf_counter() - started)
                saves.append({"stage": self.name, "dst": item["dst"], "masks": masks})
        return {"results": results, "budget": budget, "save_ref": ray.put(saves) if saves else None}

    def store(self, save: dict) -> None:
        save_masks(Path(save["dst"]), save["masks"])


def segment_video(processor, cutie, frames: np.ndarray) -> np.ndarray:
    if len(frames) == 0:
        raise RuntimeError("segment received no frames")
    indexes = list(range(0, len(frames), _STRIDE))
    keyed = _keyframes(processor, frames, indexes)
    keys = {index: mask for index, mask in zip(indexes, keyed)}
    return _propagate(cutie, frames, keys)


def _keyframes(processor, frames: np.ndarray, indexes: list[int]) -> list[np.ndarray]:
    import torch
    from PIL import Image

    masks = []
    for index in indexes:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = processor.set_image(Image.fromarray(frames[index]))
            state = processor.set_text_prompt("person", state)
        masks.append(_mask_from_state(state, frames[index].shape))
    return masks


def _mask_from_state(state, shape) -> np.ndarray:
    masks = state.get("masks")
    if masks is None or masks.numel() == 0:
        return np.zeros(shape[:2], dtype=np.uint8)
    binary = masks.detach().flatten(0, -3).any(dim=0).cpu().numpy()
    return _close(binary).astype(np.uint8)


def _propagate(cutie, frames: np.ndarray, keys: dict[int, np.ndarray]) -> np.ndarray:
    import torch

    cutie.clear_memory()
    masks = []
    with torch.inference_mode():
        for index, frame in enumerate(frames):
            given = keys[index].astype(np.uint8) if index in keys else None
            small, given, full = _fit(frame, given)
            image = torch.from_numpy(np.ascontiguousarray(small)).permute(2, 0, 1).float().cuda() / 255
            if given is not None:
                prob = cutie.step(image, torch.from_numpy(given.astype(np.int64)).cuda(), objects=[1])
            else:
                prob = cutie.step(image)
            pred = (prob.argmax(dim=0).detach().cpu().numpy() > 0).astype(np.uint8)
            if pred.shape[::-1] != full:
                pred = cv2.resize(pred, full, interpolation=cv2.INTER_NEAREST)
            masks.append(_close(pred.astype(bool)))
    return np.stack(masks)


def _fit(frame: np.ndarray, mask: np.ndarray | None):
    height, width = frame.shape[:2]
    if min(height, width) <= _SHORT:
        return frame, mask, (width, height)
    scale = _SHORT / min(height, width)
    resized_h = max(16, int(round(height * scale / 16) * 16))
    resized_w = max(16, int(round(width * scale / 16) * 16))
    frame = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    if mask is not None:
        mask = cv2.resize(mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    return frame, mask, (width, height)


def _close(mask: np.ndarray) -> np.ndarray:
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, _KERNEL).astype(bool)
