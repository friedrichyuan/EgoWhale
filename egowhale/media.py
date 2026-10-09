"""Packed masks and short RGB clips shared by the visual steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_masks(path: Path, masks: np.ndarray) -> None:
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
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    # The container's count only sizes the array; decoding still runs to `stop` or the end.
    expected = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if stop is None else stop, start + 1) - start
    if start:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, extra, kept = None, [], 0
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
    shape = (
        int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
    )
    capture.release()
    return shape


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
