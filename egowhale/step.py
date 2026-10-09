"""One pipeline stage. Serial today; the same class is a Ray actor later."""

from __future__ import annotations

import gc
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compute_lock(step):
    """One GPU section at a time. The caller copies results out after releasing it."""
    import threading

    lock = getattr(step, "_gate", None)
    if lock is None:
        lock = threading.Lock()
        step._gate = lock
    return lock

# Inpaint, depth, and render all work at this fraction of the source video.
SCALE = 0.5

GRIPPER = "action/gripper.npz"
MASKS = "visual/masks.npz"
INPAINT = "visual/inpaint.mp4"
DEPTH = "visual/depth.npz"
BASE = "action/base.json"
IK = "action/ik.npz"
PREFIX = "action/prefix.npz"
RENDER = "visual/render.npz"
COMPOSITE = "visual/composite.mp4"
QUALITY = "action/quality.json"

# Episodes longer than PIECE frames are split when they are first read, and marked in PIECES.
# Visual files then live per piece; action files and the composite stay per episode.
PIECE = 600
PIECES = "pieces.json"
PER_PIECE = (MASKS, INPAINT, DEPTH, RENDER)


def split(frames: int, limit: int) -> list[tuple[int, int]]:
    """Equal pieces of at most `limit` frames, as [start, stop) spans."""
    count = max(1, -(-frames // limit))
    edges = [round(frames * index / count) for index in range(count + 1)]
    return list(zip(edges[:-1], edges[1:]))


def plan(src: Path | str, dst: Path | str, limit: int = PIECE) -> list[tuple[int, int]]:
    """Write the episode's pieces once. A plan made with the same limit is kept."""
    path = Path(dst) / PIECES
    if path.is_file():
        marked = json.loads(path.read_text())
        if marked["limit"] == limit:
            return [tuple(span) for span in marked["pieces"]]
    frames = frame_count(src)
    spans = split(frames, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"frames": frames, "limit": limit, "pieces": spans}))
    return spans


def pieces(dst: Path | str) -> list[tuple[int, int]]:
    path = Path(dst) / PIECES
    if not path.is_file():
        raise FileNotFoundError(path)
    return [tuple(span) for span in json.loads(path.read_text())["pieces"]]


def piece_dir(dst: Path | str, index: int, count: int) -> Path:
    """An unsplit episode keeps its visual files at the top; pieces go under pieces/NN."""
    return Path(dst) if count == 1 else Path(dst) / "pieces" / f"{index:02d}"


def located(dst: Path | str, name: str) -> list[Path]:
    """Every path `name` is written to: one per piece for visual files."""
    if name not in PER_PIECE:
        return [Path(dst) / name]
    count = len(pieces(dst))
    return [piece_dir(dst, index, count) / name for index in range(count)]


def missing(dst: Path | str, names: tuple[str, ...]) -> list[str]:
    return [name for name in names if not all(path.is_file() for path in located(dst, name))]


def frame_count(src: Path | str) -> int:
    import h5py

    with h5py.File(src) as handle:
        return int(handle["transforms/camera"].shape[0])


class Step:
    """Subclass `run`. `gpus` is the only scheduling hint a later launcher needs."""

    name: str
    needs: tuple[str, ...] = ()
    makes: tuple[str, ...] = ()
    gpus: float = 0

    def run(self, src: Path, dst: Path) -> None:
        raise NotImplementedError(self.name)

    @classmethod
    def remote(cls):
        """Ray actor class. Call `run` with string paths."""
        import ray

        cached = getattr(cls, "_ray", None)
        if cached is None:
            cached = ray.remote(num_gpus=cls.gpus)(cls)
            cls._ray = cached
        return cached


def run(src: Path | str, dst: Path | str, steps: tuple[type[Step], ...]) -> None:
    """Single process, one GPU stage at a time."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    plan(src, dst)
    for cls in steps:
        if _done(dst, cls):
            print(f"skip {cls.name}")
            continue
        absent = missing(dst, cls.needs)
        if absent:
            raise FileNotFoundError(f"{cls.name} missing {absent}")
        print(f"run {cls.name}")
        cls().run(src, dst)
        absent = missing(dst, cls.makes)
        if absent:
            raise FileNotFoundError(f"{cls.name} did not write {absent}")
        if cls.gpus:
            _release_gpu()


def run_remote(src: Path | str, dst: Path | str, steps: tuple[type[Step], ...]) -> None:
    """Same order on one GPU via Ray. Overlap and multi-GPU stay a scheduler change."""
    import ray

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    plan(src, dst)
    if not ray.is_initialized():
        ray.init(num_gpus=1, ignore_reinit_error=True)
    for cls in steps:
        if _done(dst, cls):
            print(f"skip {cls.name}")
            continue
        print(f"run {cls.name}")
        if cls.gpus:
            actor = cls.remote().options(num_gpus=cls.gpus, max_concurrency=1).remote()
            ray.get(actor.run.remote(str(src), str(dst)))
        else:
            cls().run(src, dst)


def _done(dst: Path, cls: type[Step]) -> bool:
    return not missing(dst, cls.makes)


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.set_default_dtype(torch.float32)
    except ImportError:
        pass
