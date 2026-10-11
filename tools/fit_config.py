"""Fit a batch config to this host's RAM and CPUs without touching the GPU layout: python tools/fit_config.py IN OUT

The IO knobs are lowered one at a time (queue frames, readers per lane, memory reserve) until
the worst-case memory plan and the CPU count fit, and Ray's object store is capped at OBJECT_STORE_SHARE of RAM.
Every change is printed. OUT is written only when the config fits; otherwise the exit code is 1.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tempfile  # noqa: E402

import yaml  # noqa: E402

from egowhale import config  # noqa: E402
from egowhale.batch.resources import plan_memory, pools_from, total_gb  # noqa: E402

# Ray's own default object store size, as a share of RAM. A larger store leaves actors no memory and Ray refuses to start.
OBJECT_STORE_SHARE = 0.3


def _cpus() -> int:
    try:
        from ray._private.utils import get_num_cpus

        return int(get_num_cpus())
    except Exception:
        import os

        return os.cpu_count() or 1


def _need(raw: dict, path: Path) -> tuple[float, int, dict]:
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    cfg = config.load(path)
    pools, _share = pools_from(cfg)
    worst = sum(gb for _name, gb in plan_memory(cfg, pools, 1080, 1920))
    cpus = sum(pools[name] * count for name, count in cfg["io"]["cpus"].items()) + 1
    return worst + cfg["memory"]["reserve_gb"], cpus, cfg


def main() -> None:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    raw = yaml.safe_load(source.read_text())
    raw.setdefault("io", {})
    raw.setdefault("memory", {})
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp()) / target.name
    have_gb, have_cpus = total_gb(), _cpus()
    print(f"host: {have_gb:.0f} GiB RAM, {have_cpus} CPUs for Ray")
    need_gb, need_cpus, cfg = _need(raw, scratch)
    print(f"{source.name} as written: needs {need_gb:.0f} GiB (worst case + reserve), {need_cpus} CPUs")
    steps = [
        ("io", "queue_frames", lambda value: max(cfg["piece_frames"], value - 600)),
        ("io", "readers_per_lane", lambda value: max(1, value - 1)),
        ("memory", "reserve_gb", lambda value: max(16, value // 2)),
    ]
    for section, key, lower in steps:
        while need_gb > have_gb or need_cpus > have_cpus:
            current = raw[section].get(key, cfg[section][key])
            lowered = lower(current)
            if lowered == current:
                break
            raw[section][key] = lowered
            need_gb, need_cpus, cfg = _need(raw, scratch)
            print(f"  {section}.{key}: {current} -> {lowered}  now {need_gb:.0f} GiB, {need_cpus} CPUs")
    store = cfg["memory"]["object_store_gb"]
    cap = int(have_gb * OBJECT_STORE_SHARE)
    if store == 0 or store > cap:
        raw["memory"]["object_store_gb"] = cap
        print(f"  memory.object_store_gb: {store} -> {cap}  ({OBJECT_STORE_SHARE:.0%} of RAM)")
    if need_gb > have_gb or need_cpus > have_cpus:
        print(f"does not fit: needs {need_gb:.0f} GiB and {need_cpus} CPUs")
        sys.exit(1)
    target.write_text(yaml.safe_dump(raw, sort_keys=False))
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
