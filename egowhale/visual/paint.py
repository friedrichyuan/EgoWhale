"""ProPainter weights and batched inpainting, on the mask box of each piece."""

from __future__ import annotations

import os

import numpy as np

from egowhale import thirdparty
from egowhale.media import CHUNK, MASK_DILATION, SCALE, dilate_masks, read_chunks, resize_frames

_URL = "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/"


def _libs():
    thirdparty.use("propainter")
    from inference_propainter import get_ref_index
    from model.misc import get_device
    from model.modules.flow_comp_raft import RAFT_bi
    from model.propainter import InpaintGenerator
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from utils.download_util import load_file_from_url

    return {
        "get_ref_index": get_ref_index,
        "get_device": get_device,
        "RAFT_bi": RAFT_bi,
        "InpaintGenerator": InpaintGenerator,
        "RecurrentFlowCompleteNet": RecurrentFlowCompleteNet,
        "load_file_from_url": load_file_from_url,
    }


def load_models(weight_dir: str, device=None):
    """RAFT, flow completion, and the painter. They stay on the inpaint actor."""
    libs = _libs()
    device = device or libs["get_device"]()
    os.makedirs(weight_dir, exist_ok=True)
    raft_path = libs["load_file_from_url"](
        url=_URL + "raft-things.pth", model_dir=weight_dir, progress=False, file_name=None
    )
    raft = libs["RAFT_bi"](raft_path, device)
    flow_path = libs["load_file_from_url"](
        url=_URL + "recurrent_flow_completion.pth", model_dir=weight_dir, progress=False, file_name=None
    )
    flow = libs["RecurrentFlowCompleteNet"](flow_path)
    for parameter in flow.parameters():
        parameter.requires_grad = False
    flow.to(device).eval()
    paint_path = libs["load_file_from_url"](
        url=_URL + "ProPainter.pth", model_dir=weight_dir, progress=False, file_name=None
    )
    painter = libs["InpaintGenerator"](model_path=paint_path).to(device).eval()
    return {"raft": raft, "flow": flow, "painter": painter, "device": device, "half": False}


def prepare_clip(frames: np.ndarray, masks: np.ndarray, resize_ratio: float = SCALE, mask_dilation: int = MASK_DILATION) -> dict:
    """Resize and dilate on the CPU dataloader. Pixels stay uint8 until they reach the GPU."""
    import torch

    height, width = frames.shape[1], frames.shape[2]
    out_size, process = _sizes(width, height, resize_ratio)
    return {
        "frames": torch.from_numpy(resize_frames(frames, process)).permute(0, 3, 1, 2).unsqueeze(0).contiguous(),
        "masks": dilate_masks(masks, process, mask_dilation),
        "out_size": out_size,
        "key": (*process, len(frames)),
    }


def crop_box(masks: np.ndarray, margin: float = 0.05) -> tuple[int, int, int, int]:
    """The clip's mask box plus a margin, as y0, y1, x0, x1 on 16-pixel edges. One box keeps flow coordinates fixed."""
    height, width = masks.shape[1:]
    ys, xs = np.where(masks.any(0))
    if len(ys) == 0:
        return 0, height, 0, width
    pad = int(margin * max(height, width))
    y0 = max(0, (ys.min() - pad) // 16 * 16)
    x0 = max(0, (xs.min() - pad) // 16 * 16)
    y1 = min(height, -(-(ys.max() + 1 + pad) // 16) * 16)
    x1 = min(width, -(-(xs.max() + 1 + pad) // 16) * 16)
    return int(y0), int(y1), int(x0), int(x1)


def prepare_crop(video, start: int, stop: int, masks: np.ndarray, resize_ratio: float = SCALE, chunk: int = CHUNK) -> dict:
    """Decode frames [start, stop) a chunk at a time, keeping only the resized mask box.

    The full frames are never held for the whole piece, and the background is not kept: the writer
    decodes it again when it pastes the painted box back.
    """
    import torch

    y0, y1, x0, x1 = crop_box(masks)
    out_size, process = _sizes(x1 - x0, y1 - y0, resize_ratio)
    resized = np.empty((stop - start, process[1], process[0], 3), dtype=np.uint8)
    count = 0
    fps = 30.0
    for fps, block in read_chunks(video, start, stop, chunk):
        resized[count : count + len(block)] = resize_frames(np.ascontiguousarray(block[:, y0:y1, x0:x1]), process)
        count += len(block)
    if count != len(masks):
        raise ValueError(f"{count} frames vs {len(masks)} masks")
    return {
        "frames": torch.from_numpy(resized).permute(0, 3, 1, 2).unsqueeze(0).contiguous(),
        "masks": dilate_masks(masks[:, y0:y1, x0:x1], process),
        "out_size": out_size,
        "key": (*process, count),
        "box": (int(y0 * resize_ratio), int(x0 * resize_ratio)),
        "fps": float(fps),
    }


def write_painted(
    path, video, start: int, stop: int, painted: np.ndarray, box: tuple[int, int] | None, resize_ratio: float = SCALE, chunk: int = CHUNK
) -> None:
    """Paste the painted box into the resized source frames and write the video, a chunk at a time.

    With no box the painted frames are whole frames and are written as they are; the source is not decoded.
    """
    from egowhale.media import RgbWriter, video_shape

    if box is None:
        with RgbWriter(path, video_shape(video)[3]) as writer:
            writer.write(painted)
        return
    top, left = box
    writer = None
    done = 0
    try:
        for fps, block in read_chunks(video, start, stop, chunk):
            if writer is None:
                writer = RgbWriter(path, fps)
            height, width = block.shape[1:3]
            background = resize_frames(block, (int(width * resize_ratio), int(height * resize_ratio)))
            part = painted[done : done + len(background)]
            background[: len(part), top : top + part.shape[1], left : left + part.shape[2]] = part
            writer.write(background)
            done += len(background)
    finally:
        if writer is not None:
            writer.close()


def _sizes(width: int, height: int, resize_ratio: float) -> tuple[tuple[int, int], tuple[int, int]]:
    """Output size, and ProPainter's process size: the output size rounded down to a multiple of 8."""
    out_size = (int(width * resize_ratio), int(height * resize_ratio))
    return out_size, (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)


def paint_clips(models, clips: list[dict], ref_stride=10, neighbor_length=10, subvideo_length=80, raft_iter=20, fp16=True):
    """One forward per group of clips that share process size and length."""
    import torch

    buckets: dict[tuple, list] = {}
    for index, clip in enumerate(clips):
        buckets.setdefault(clip["key"], []).append(index)
    painted = [None] * len(clips)
    for indexes in buckets.values():
        group = [clips[index] for index in indexes]
        outputs = _forward(
            models,
            torch.cat([clip["frames"] for clip in group], dim=0),
            torch.cat([clip["masks"] for clip in group], dim=0),
            [clip["out_size"] for clip in group],
            ref_stride=ref_stride,
            neighbor_length=neighbor_length,
            subvideo_length=subvideo_length,
            raft_iter=raft_iter,
            fp16=fp16,
        )
        for index, frames in zip(indexes, outputs):
            painted[index] = frames
    return painted


def _forward(models, frames, masks, out_sizes, ref_stride, neighbor_length, subvideo_length, raft_iter, fp16):
    import cv2
    import torch

    device = models["device"]
    # ProPainter's to_tensors maps pixels to [-1, 1] on the CPU. GPU division rounds differently in the last bit,
    # so the 256 CPU results are looked up instead of recomputed.
    table = (torch.arange(256, dtype=torch.float32).div(255) * 2 - 1).to(device)
    pixels = frames
    frames = table[pixels.to(device).long()]
    masks = masks.to(device).float()
    flow_masks = masks
    batch, video_length, _, height, width = frames.shape
    raft = models["raft"]
    flow = models["flow"]
    painter = models["painter"]
    get_ref_index = _libs()["get_ref_index"]

    with torch.no_grad():
        if width <= 640:
            short_clip = 12
        elif width <= 720:
            short_clip = 8
        elif width <= 1280:
            short_clip = 4
        else:
            short_clip = 2
        if video_length > short_clip:
            forward_flows, backward_flows = [], []
            for start in range(0, video_length, short_clip):
                end = min(video_length, start + short_clip)
                window = frames[:, start:end] if start == 0 else frames[:, start - 1 : end]
                flows_f, flows_b = raft(window, iters=raft_iter)
                forward_flows.append(flows_f)
                backward_flows.append(flows_b)
            flows = (torch.cat(forward_flows, dim=1), torch.cat(backward_flows, dim=1))
        else:
            flows = raft(frames, iters=raft_iter)

        use_half = bool(fp16) and getattr(device, "type", "") == "cuda"
        if use_half:
            frames = frames.half()
            flow_masks = flow_masks.half()
            masks = masks.half()
            flows = (flows[0].half(), flows[1].half())
        # The shared weights follow this call's precision, so a later fp32 call does not meet half weights.
        if models["half"] != use_half:
            cast = (lambda module: module.half()) if use_half else (lambda module: module.float())
            models["flow"] = cast(flow)
            models["painter"] = cast(painter)
            flow = models["flow"]
            painter = models["painter"]
            models["half"] = use_half

        flow_length = flows[0].size(1)
        if flow_length > subvideo_length:
            pred_f, pred_b = [], []
            pad = 5
            for start in range(0, flow_length, subvideo_length):
                left = max(0, start - pad)
                right = min(flow_length, start + subvideo_length + pad)
                pad_left = max(0, start) - left
                pad_right = right - min(flow_length, start + subvideo_length)
                pair = (flows[0][:, left:right], flows[1][:, left:right])
                completed, _edges = flow.forward_bidirect_flow(pair, flow_masks[:, left : right + 1])
                completed = flow.combine_flow(pair, completed, flow_masks[:, left : right + 1])
                pred_f.append(completed[0][:, pad_left : right - left - pad_right])
                pred_b.append(completed[1][:, pad_left : right - left - pad_right])
            completed_flows = (torch.cat(pred_f, dim=1), torch.cat(pred_b, dim=1))
        else:
            completed_flows, _edges = flow.forward_bidirect_flow(flows, flow_masks)
            completed_flows = flow.combine_flow(flows, completed_flows, flow_masks)

        masked = frames * (1 - masks)
        prop_length = min(100, subvideo_length)
        if video_length > prop_length:
            updated_frames, updated_masks = [], []
            pad = 10
            for start in range(0, video_length, prop_length):
                left = max(0, start - pad)
                right = min(video_length, start + prop_length + pad)
                pad_left = max(0, start) - left
                pad_right = right - min(video_length, start + prop_length)
                count = masks[:, left:right].size(1)
                pair = (completed_flows[0][:, left : right - 1], completed_flows[1][:, left : right - 1])
                propagated, local_masks = painter.img_propagation(masked[:, left:right], pair, masks[:, left:right], "nearest")
                filled = frames[:, left:right] * (1 - masks[:, left:right]) + propagated.view(batch, count, 3, height, width) * masks[:, left:right]
                updated_frames.append(filled[:, pad_left : right - left - pad_right])
                updated_masks.append(local_masks.view(batch, count, 1, height, width)[:, pad_left : right - left - pad_right])
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            propagated, local_masks = painter.img_propagation(masked, completed_flows, masks, "nearest")
            updated_frames = frames * (1 - masks) + propagated.view(batch, video_length, 3, height, width) * masks
            updated_masks = local_masks.view(batch, video_length, 1, height, width)

    stride = neighbor_length // 2
    ref_num = subvideo_length // ref_stride if video_length > subvideo_length else -1
    # The uint8 input, channels last, is what unmasked pixels keep in the output.
    base = pixels.permute(0, 1, 3, 4, 2).to(device).contiguous()
    composite = torch.zeros_like(base)
    filled = torch.zeros(video_length, dtype=torch.bool, device=device)
    for start in range(0, video_length, stride):
        neighbors = list(range(max(0, start - stride), min(video_length, start + stride + 1)))
        refs = get_ref_index(start, neighbors, video_length, ref_stride, ref_num)
        selected = neighbors + refs
        with torch.no_grad():
            predicted = painter(
                updated_frames[:, selected],
                (completed_flows[0][:, neighbors[:-1]], completed_flows[1][:, neighbors[:-1]]),
                masks[:, selected],
                updated_masks[:, selected],
                len(neighbors),
            )
            predicted = (predicted.float() + 1) / 2
            predicted = predicted.permute(0, 1, 3, 4, 2) * 255
            binary = (masks[:, neighbors].float().permute(0, 1, 3, 4, 2) > 0.5).to(torch.uint8)
            image = (predicted * binary + base[:, neighbors] * (1 - binary)).to(torch.uint8)
            previous = filled[neighbors].view(1, -1, 1, 1, 1)
            blended = (composite[:, neighbors].float() * 0.5 + image.float() * 0.5).to(torch.uint8)
            composite[:, neighbors] = torch.where(previous, blended, image)
            filled[neighbors] = True

    if not bool(filled.all()):
        raise RuntimeError("ProPainter left a frame empty")
    composite = composite.cpu().numpy()
    outputs = []
    for item, out_size in enumerate(out_sizes):
        frames_out = []
        for frame in composite[item]:
            if (frame.shape[1], frame.shape[0]) != tuple(out_size):
                frame = cv2.resize(frame, tuple(out_size), interpolation=cv2.INTER_CUBIC)
            frames_out.append(frame)
        outputs.append(np.stack(frames_out))
    return outputs
