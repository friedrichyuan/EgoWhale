"""Batch configuration: experiment name, output directory, episodes, node pools, and memory limits.

A file in cfgs/ only lists what it changes; everything else comes from DEFAULTS.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from egowhale.step import PIECE, ROOT

DEFAULTS = {
    "name": "run",
    # A .txt list of HDF5 paths, an HDF5 file, or a dataset directory; or a list of those.
    "episodes": None,
    # Results go to <out>/<name>/. Relative paths are taken from the repository root.
    "out": "outputs",
    # 0 runs every episode.
    "limit": 0,
    # Longer episodes are split into equal pieces of at most this many frames.
    "piece_frames": PIECE,
    "nodes": {
        # SAM 3 + Cutie. Actors on one card share it evenly.
        "segment": {"cards": 1, "actors_per_card": 1},
        # ProPainter, one actor per card.
        "inpaint": {"cards": 1},
        # DA3-GIANT. `window` views share one DA3 call; `frames_per_forward` frames go in one forward.
        "depth": {"cards": 1, "window": 1, "frames_per_forward": 96},
        # cuRobo. `ride: inpaint|depth` puts `actors_per_card` actors on every card of that node,
        # each holding `gpu_share` of it; otherwise the action node gets `cards` cards of its own.
        "action": {"ride": None, "cards": 1, "actors_per_card": 1, "gpu_share": 0.1},
        # MuJoCo EGL render actors. They draw on the action cards.
        "render": {"actors": 4},
        # Composite and curate actors.
        "cpu": {"actors": 8},
    },
    "io": {
        # CPU decoders for each visual GPU actor.
        "readers_per_lane": 2,
        # Pieces one visual GPU actor may hold: decoding, decoded, and on the GPU.
        "queue_per_lane": 4,
        # Frames one visual GPU actor may hold across those pieces. One piece always fits.
        "queue_frames": 1200,
        # CPUs each actor reserves for its whole life. Ray queues actors silently when they do not fit.
        "cpus": {"cpu": 1, "render": 1, "feed": 2, "write": 1, "action_read": 1, "action_write": 1},
    },
    "memory": {
        # Episodes open at once. Open episodes cost disk, not RAM; this only sets how far fast nodes run ahead.
        "inflight": 64,
        # No new piece is decoded while the host has less than this free.
        "reserve_gb": 64,
        # Ray shared memory. Beyond it objects spill to disk instead of growing RAM. 0 keeps Ray's default.
        "object_store_gb": 120,
    },
}


def load(path: Path | str, **overrides) -> dict:
    """Defaults, then the file, then non-None keyword overrides of top-level keys."""
    path = Path(path)
    given = yaml.safe_load(path.read_text()) or {}
    config = _merge(copy.deepcopy(DEFAULTS), given, "")
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if not config["episodes"]:
        raise SystemExit(f"{path}: no episodes")
    action = config["nodes"]["action"]
    if action["ride"] not in (None, "inpaint", "depth"):
        raise SystemExit(f"{path}: nodes.action.ride must be inpaint, depth, or empty")
    if action["ride"] and action["actors_per_card"] * action["gpu_share"] >= 1:
        raise SystemExit(f"{path}: riding action actors would take the whole card")
    config["out"] = str(_resolve(config["out"]) / config["name"])
    config["source"] = str(path.resolve())
    return config


def episodes(config: dict) -> list[Path]:
    entries = config["episodes"]
    found: list[Path] = []
    for entry in entries if isinstance(entries, list) else [entries]:
        path = _resolve(entry)
        if path.suffix == ".txt":
            found.extend(Path(line.strip()).expanduser() for line in path.read_text().splitlines() if line.strip())
        elif path.is_dir():
            found.extend(sorted(item for item in path.rglob("*.hdf5") if item.with_suffix(".mp4").is_file()))
        else:
            found.append(path)
    if config["limit"]:
        found = found[: config["limit"]]
    return [path.resolve() for path in found]


def _resolve(value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _merge(base: dict, given: dict, where: str) -> dict:
    for key, value in given.items():
        if key not in base:
            raise SystemExit(f"unknown config key {where}{key}")
        if isinstance(base[key], dict) and base[key] and isinstance(value, dict):
            _merge(base[key], value, f"{where}{key}.")
        else:
            base[key] = value
    return base
