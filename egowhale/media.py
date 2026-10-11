"""Video, mask, and depth files shared by the stages, and the sizes the visual stages work at."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Inpaint, depth, and render all work at this fraction of the source video.
SCALE = 0.5
# EgoDex records at 30 fps.
FPS = 30.0
# Frames decoded at once by chunked readers and writers. Full-size frames never pile up beyond this.
CHUNK = 32
# Pixels a person mask grows by before it is painted out, at SCALE.
MASK_DILATION = 4
_RESIZE_THREADS = 8


def save_masks(path: Path, masks: np.ndarray) -> None:
    path = Path(path)
    masks = np.asarray(masks, dtype=bool)
    packed = np.packbits(masks.reshape(masks.shape[0], -1), axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, packed=packed, shape=np.array(masks.shape, dtype=np.int32))


def load_masks(path: Path) -> np.ndarray:
    with np.load(path) as data:
        shape = tuple(int(value) for value in data["shape"])
        packed = data["packed"]
    flat = np.unpackbits(packed, axis=1)[:, : shape[1] * shape[2]]
    return flat.reshape(shape).astype(bool)


def read_rgb(path: Path, start: int = 0, stop: int | None = None) -> tuple[np.ndarray, float]:
    """Frames [start, stop), decoded straight into one array so a long piece is held once."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or FPS)
    # The container's count only sizes the array; decoding still runs to `stop` or the end.
    expected = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if stop is None else stop, start + 1) - start
    if start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, extra, kept = None, [], 0
    try:
        while stop is None or start + kept < stop:
            ok, frame = capture.read()
            if not ok:
                break
            if frames is None:
                frames = np.empty((expected, *frame.shape), dtype=np.uint8)
            if kept < len(frames):
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=frames[kept])
            else:
                extra.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            kept += 1
    finally:
        capture.release()
    if frames is None:
        raise ValueError(f"no frames in {path} from {start}")
    if extra:
        return np.concatenate([frames, np.stack(extra)]), fps
    return (frames if kept == len(frames) else frames[:kept].copy()), fps


def read_frames(path: Path, indices) -> np.ndarray:
    """Only the listed frames, in order. The rest are skipped without color conversion."""
    import cv2

    wanted = sorted(set(int(index) for index in indices))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    frames = []
    position = 0
    try:
        for index in wanted:
            while position < index:
                if not capture.grab():
                    break
                position += 1
            ok, frame = capture.read()
            if not ok:
                break
            position += 1
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"no frames in {path}")
    return np.stack(frames)


def video_shape(path: Path) -> tuple[int, int, int, float]:
    """Frame count, height, width, fps from the container."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            float(capture.get(cv2.CAP_PROP_FPS) or FPS),
        )
    finally:
        capture.release()


def frame_size(path: Path) -> tuple[int, int]:
    """Height and width of a video's frames. A file that is not a video raises FileNotFoundError."""
    _count, height, width, _fps = video_shape(path)
    if height <= 0 or width <= 0:
        raise FileNotFoundError(path)
    return height, width


def read_chunks(path: Path, start: int, stop: int, chunk: int = CHUNK):
    """(fps, frames) blocks of at most `chunk` RGB frames from [start, stop)."""
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or FPS)
    if start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    position = start
    try:
        while position < stop:
            block = []
            while len(block) < chunk and position < stop:
                ok, frame = capture.read()
                if not ok:
                    break
                block.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                position += 1
            if not block:
                break
            yield fps, np.stack(block)
            if len(block) < chunk and position < stop:
                break
    finally:
        capture.release()


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """PIL's default resize to (width, height), one thread per frame. PIL drops the GIL while resampling."""
    from concurrent.futures import ThreadPoolExecutor

    from PIL import Image

    with ThreadPoolExecutor(_RESIZE_THREADS) as pool:
        return np.stack(list(pool.map(lambda frame: np.asarray(Image.fromarray(frame).resize(size)), frames)))


def dilate_masks(masks: np.ndarray, size: tuple[int, int], iterations: int = MASK_DILATION):
    """Masks resized to (width, height) and grown, as a uint8 tensor shaped (1, frames, 1, height, width)."""
    import cv2
    import scipy.ndimage
    import torch

    out = np.empty((len(masks), size[1], size[0]), dtype=np.uint8)
    for index, mask in enumerate(masks):
        image = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        if iterations > 0:
            out[index] = scipy.ndimage.binary_dilation(image, iterations=iterations)
        else:
            out[index] = image > 0
    return torch.from_numpy(out).unsqueeze(1).unsqueeze(0)


def save_depth(path: Path, depth: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, depth=np.asarray(depth, dtype=np.float32))


class RgbWriter:
    """An mp4 written a batch of frames at a time. Opens on the first batch."""

    def __init__(self, path: Path, fps: float):
        self.path = Path(path)
        self.fps = fps
        self._writer = None

    def write(self, frames: np.ndarray) -> None:
        import cv2

        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frames.shape[1:3]
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
            if not self._writer.isOpened():
                raise RuntimeError(f"cannot write {self.path}")
        for frame in frames:
            self._writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def write_rgb(path: Path, frames: np.ndarray, fps: float) -> None:
    with RgbWriter(path, fps) as writer:
        writer.write(frames)
