"""Batch configuration: experiment name, output directory, episodes, node pools, and memory limits.

A file in cfgs/ only lists what it changes; everything else comes from DEFAULTS.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from egowhale.paths import ROOT
from egowhale.pieces import PIECE


class ConfigError(ValueError):
    """A config that cannot run. The command line prints it and exits."""


DEFAULTS = {
    "name": "run",
    "robot": "arx",
    "approach": True,
    # A .txt list of HDF5 paths, an HDF5 file, or a dataset directory; or a list of those.
    "episodes": None,
    # Results go to <out>/<name>/. Relative paths are taken from the repository root.
    "out": "outputs",
    # 0 runs every episode.
    "limit": 0,
    # "list" keeps the listed order; "shuffle" mixes long and short episodes evenly, with `seed`.
    "order": "list",
    "seed": 0,
    # Longer episodes are split into equal pieces of at most this many frames.
    "piece_frames": PIECE,
    "nodes": {
        # SAM 3 + Cutie. Actors on one card share it evenly.
        "segment": {"cards": 1, "actors_per_card": 1},
        # One actor per card. `model`: sttn (fast) or propainter (slower, sharper).
        "inpaint": {"cards": 1, "model": "sttn"},
        # DA3-GIANT. `window` views share one DA3 call; `frames_per_forward` frames go in one forward.
        "depth": {"cards": 1, "window": 1, "frames_per_forward": 96},
        # cuRobo. `ride: inpaint|depth` puts `actors_per_card` actors on every card of that node,
        # each holding `gpu_share` of it; otherwise the action node gets `cards` cards of its own.
        "action": {"ride": None, "cards": 1, "actors_per_card": 1, "gpu_share": 0.1, "batch_size": 0, "ik_method": None},
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
        # Output batches reserved at GPU submission until their writer finishes; 0 keeps writes unbounded.
        "write_slots_per_lane": 0,
        # CPUs each actor reserves for its whole life. Ray queues actors silently when they do not fit.
        "cpus": {"cpu": 1, "render": 1, "feed": 2, "write": 1, "action_read": 1, "action_write": 1},
    },
    "node_bench": {"workers": None, "warmup": 1, "inputs_root": None},
    "memory": {
        # Episodes open at once. Open episodes cost disk, not RAM; this only sets how far fast nodes run ahead.
        "inflight": 64,
        # No new piece is decoded while the host has less than this free.
        "reserve_gb": 64,
        # Ray shared memory. Beyond it objects spill to disk instead of growing RAM. 0 keeps Ray's default.
        "object_store_gb": 120,
    },
}


def read_config(path: Path | str) -> dict:
    given = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(given, dict):
        raise ConfigError(f"{path}: expected a YAML mapping")
    return _merge(copy.deepcopy(DEFAULTS), given, "")


def ik_options(action, robot):
    size = action["batch_size"]
    if type(size) is not int or size < 0:
        raise ConfigError("nodes.action.batch_size must be a nonnegative integer")
    method = action["ik_method"]
    if method is None:
        method = "base_ik_seq" if size else "base_ik"
    if method == "base_ik":
        return method, 0, 0
    if method != "base_ik_seq":
        raise ConfigError("nodes.action.ik_method must be base_ik or base_ik_seq")
    from egowhale.action.base_ik_seq import episode_capacity

    try:
        return method, size, episode_capacity(robot, size)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc


def load(path: Path | str, **overrides) -> dict:
    """Defaults, then the file, then non-None keyword overrides of top-level keys."""
    path = Path(path)
    config = read_config(path)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    if type(config["approach"]) is not bool:
        raise ConfigError("approach must be true or false")
    if config["robot"] not in ("arx", "piper", "dual_piper"):
        raise ConfigError("robot must be arx or piper")
    if not config["episodes"]:
        raise ConfigError(f"{path}: no episodes")
    slots = config["io"]["write_slots_per_lane"]
    if type(slots) is not int or slots < 0:
        raise ConfigError(f"{path}: io.write_slots_per_lane must be a nonnegative integer")
    from egowhale.visual.inpaint import MODELS

    if config["nodes"]["inpaint"]["model"] not in MODELS:
        raise ConfigError(f"{path}: nodes.inpaint.model must be one of {', '.join(MODELS)}")
    action = config["nodes"]["action"]
    method, size, capacity = ik_options(action, config["robot"])
    action.update(ik_method=method, batch_size=size)
    if capacity > config["memory"]["inflight"]:
        raise ConfigError("memory.inflight must be at least the episode batch capacity")
    if action["ride"] not in (None, "inpaint", "depth"):
        raise ConfigError(f"{path}: nodes.action.ride must be inpaint, depth, or empty")
    if action["ride"] and action["actors_per_card"] * action["gpu_share"] >= 1:
        raise ConfigError(f"{path}: riding action actors would take the whole card")
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
    if config["order"] == "shuffle":
        import random

        random.Random(config["seed"]).shuffle(found)
    elif config["order"] != "list":
        raise ConfigError(f"order must be list or shuffle, not {config['order']}")
    if config["limit"]:
        found = found[: config["limit"]]
    return [path.resolve() for path in found]


def _resolve(value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _merge(base: dict, given: dict, where: str) -> dict:
    for key, value in given.items():
        if key not in base:
            raise ConfigError(f"unknown config key {where}{key}")
        if isinstance(base[key], dict) and base[key] and isinstance(value, dict):
            _merge(base[key], value, f"{where}{key}.")
        else:
            base[key] = value
    return base
