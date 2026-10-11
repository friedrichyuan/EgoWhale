"""Episodes longer than a piece are split when they are first read, and the split is kept in pieces.json.

Visual files then live per piece; action files and the composite stay per episode.
"""

from __future__ import annotations

import json
from pathlib import Path

from egowhale.paths import PER_PIECE, PIECES

# Default longest piece, in frames.
PIECE = 600


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
    from egowhale.media import video_shape

    # Later stages read as many video frames as the HDF5 has poses; a shorter video would be cut silently.
    video = video_shape(Path(src).with_suffix(".mp4"))[0]
    if video != frames:
        raise ValueError(f"{src}: video has {video} frames, HDF5 has {frames}")
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
