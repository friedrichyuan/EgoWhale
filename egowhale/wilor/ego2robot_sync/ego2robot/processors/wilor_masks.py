"""SAM3 hand-mask propagation shared by both WiLoR entry points."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ego2robot.utils.progress import report_progress


def build_predictor(checkpoint):
    from sam3.model_builder import build_sam3_video_predictor

    options = {"checkpoint_path": checkpoint} if checkpoint else {}
    return build_sam3_video_predictor(gpus_to_use=[0], **options)


def infer_masks(predictor, frames, masks):
    count, height, width = masks.shape
    images = [Image.fromarray(frame) for frame in frames]
    session = predictor.handle_request(dict(type="start_session", resource_path=images,
                                             offload_video_to_cpu=True))["session_id"]
    try:
        for anchor in dict.fromkeys((count // 2, 0, count // 4, 3 * count // 4, count - 1)):
            response = predictor.handle_request(dict(type="add_prompt", session_id=session,
                                                     frame_index=anchor, text="hand"))
            if len(response["outputs"].get("out_obj_ids", [])):
                break
        else:
            return

        seen = set()
        for response in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session)):
            frame = response["frame_index"]
            seen.add(frame)
            values = response["outputs"].get("out_binary_masks")
            if values is not None:
                mask = values.any(dim=0).cpu().numpy() if hasattr(values, "cpu") else np.asarray(values).any(axis=0)
                if mask.shape != (height, width):
                    mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                masks[frame] = mask
            if len(seen) % 30 == 0 or len(seen) == count:
                report_progress("SAM3 手部掩码", len(seen), count)
    finally:
        predictor.handle_request(dict(type="close_session", session_id=session))


def propagate_masks(predictor, source, save_path, shape=None, max_frames=0):
    """Use the same SAM3 propagation for a clip or bounded overlapping video windows."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count, height, width = source.shape[:3] if shape is None else shape
    masks = np.lib.format.open_memmap(path, mode="w+", dtype=bool, shape=(count, height, width))
    try:
        if shape is None:
            infer_masks(predictor, source, masks)
            return
        from egowhale.wilor.parallel_stages import rgb_frames
        from contextlib import closing
        from itertools import islice

        size = min(max_frames or count, count)
        overlap = min(16, size // 4)
        with closing(rgb_frames(source)) as decoder:
            frames = list(islice(decoder, size))
            start = 0
            while frames:
                stop = start + len(frames)
                chunk = np.zeros((len(frames), height, width), dtype=bool)
                report_progress("SAM3 窗口", stop, count, f"帧 [{start}, {stop})")
                infer_masks(predictor, frames, chunk)
                # Keep the left window through the midpoint, and the right window after it.
                keep = overlap // 2 if start else 0
                masks[start + keep:stop] = chunk[keep:]
                if stop == count:
                    if next(decoder, None) is not None:
                        raise ValueError("SAM3 decoded more frames than the prepared manifest")
                    break
                tail = frames[-overlap:] if overlap else []
                start = stop - len(tail)
                more = list(islice(decoder, size - len(tail)))
                if not more:
                    raise ValueError("SAM3 decoded fewer frames than the prepared manifest")
                frames = tail + more
            if not frames or stop != count:
                raise ValueError("SAM3 did not cover the prepared timeline")
    finally:
        del masks
