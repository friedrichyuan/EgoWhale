"""How many actors each pool gets, and the checks that the cluster and host can hold them."""

from __future__ import annotations

import os

from egowhale.batch.dag import GPU_POOLS, VISUAL
from egowhale.config import ConfigError
from egowhale.media import CHUNK

# Lanes a visual GPU actor keeps in flight; results waiting for the writer are sized by it.
INFLIGHT = 2
# Host GiB per worker process for Python, CUDA, and weights: GPU workers, then CPU workers.
GPU_WORKER_GB = 4
CPU_WORKER_GB = 0.5


def pools_from(cfg: dict) -> tuple[dict[str, int], dict[str, float]]:
    """Actor counts per pool, and the GPU share of every pool that does not hold whole cards."""
    nodes = cfg["nodes"]
    segment = nodes["segment"]["cards"] * nodes["segment"]["actors_per_card"]
    action = nodes["action"]
    ride = action["ride"]
    if ride:
        actors = nodes[ride]["cards"] * action["actors_per_card"]
    else:
        actors = action["cards"] * action["actors_per_card"]
    visual = segment + nodes["inpaint"]["cards"] + nodes["depth"]["cards"]
    readers = cfg["io"]["readers_per_lane"]
    pools = {
        "segment": segment,
        "inpaint": nodes["inpaint"]["cards"],
        "depth": nodes["depth"]["cards"],
        "action": actors,
        "cpu": nodes["cpu"]["actors"],
        "render": nodes["render"]["actors"],
        # Readers and writers are handed out per visual lane, in VISUAL order.
        "feed": visual * readers,
        "write": visual,
        "action_read": actors,
        "action_write": actors,
    }
    share = {}
    if ride:
        # The ridden node is placed first among fractional pools, so each of its actors takes its own card.
        share[ride] = round(1 - action["actors_per_card"] * action["gpu_share"], 4)
    if nodes["segment"]["actors_per_card"] > 1:
        share["segment"] = 1.0 / nodes["segment"]["actors_per_card"]
    share["action"] = action["gpu_share"] if ride else 1.0 / action["actors_per_card"]
    if share["action"] == 1.0:
        del share["action"]
    return pools, share


def available_gb() -> float | None:
    """MemAvailable from /proc/meminfo; None where it does not exist."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        return None
    return None


def total_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def plan_memory(cfg: dict, pools: dict[str, int], height: int, width: int) -> list[tuple[str, float]]:
    """Worst-case host memory in GiB by holder, for full-size pieces of a height x width source.

    Bytes per frame: a decoded frame, and the half-size frames inpaint, depth, and render work at.
    """
    frame = height * width * 3
    half = (height // 2) * (width // 2)
    piece = cfg["piece_frames"]
    queue = max(cfg["io"]["queue_frames"], piece)
    # What a queued piece costs per frame. Decoding pieces also hold the decoded frames.
    held = {
        "segment": frame,
        # Frames and masks at half size: STTN paints whole frames, ProPainter's crop is at most one.
        "inpaint": half * 3 + half,
        # DA3 input at 336 on the long side, float32.
        "depth": 3 * 4 * 336 * 336 * height // width,
    }
    readers = cfg["io"]["readers_per_lane"]
    lanes = {stage: pools[stage] for stage in VISUAL}
    rows = []
    for stage in VISUAL:
        # Inpaint decodes a chunk at a time; the others decode the whole piece first.
        decoded = CHUNK if stage == "inpaint" else piece
        decoding = readers * (decoded * frame + piece * held[stage])
        rows.append((f"{stage} queues", lanes[stage] * (queue * held[stage] + decoding) / 2**30))
    # Results waiting for the writer: masks are bool at full size, the painted frames, depth float32 at half.
    saved = {"segment": height * width, "inpaint": half * 3, "depth": half * 4}
    rows.append(("visual results", sum(lanes[stage] * INFLIGHT * piece * saved[stage] for stage in VISUAL) / 2**30))
    # Render holds rgb, part, gripper mask, and depth at half size.
    rows.append(("render", pools["render"] * piece * half * (3 + 1 + 1 + 4) / 2**30))
    # Composite holds a piece of background, render, scene depth, robot depth, masks at full and half size, and the output.
    rows.append(("composite", pools["cpu"] * piece * (half * (3 + 3 + 1 + 1 + 4 + 4 + 1) + height * width) / 2**30))
    gpu_workers = sum(pools[name] for name in GPU_POOLS)
    cpu_workers = sum(pools[name] for name in ("render", "cpu", "feed", "write", "action_read", "action_write"))
    rows.append(("worker processes", gpu_workers * GPU_WORKER_GB + cpu_workers * CPU_WORKER_GB))
    return rows


def preflight(cfg: dict, pools: dict[str, int], share: dict[str, float], height: int, width: int) -> None:
    """Refuse a layout the Ray cluster or the host cannot hold, before any actor starts."""
    import ray

    have = int(ray.cluster_resources().get("GPU", 0))
    need = round(sum(pools[name] * share.get(name, 1.0) for name in GPU_POOLS))
    if need > have:
        raise ConfigError(f"GPU pools ask for {need} devices, cluster has {have}")
    # GPU actors need one free CPU while they are being created.
    want = sum(pools[name] * count for name, count in cfg["io"]["cpus"].items()) + 1
    have = int(ray.cluster_resources().get("CPU", 0))
    if want > have:
        raise ConfigError(f"CPU pools ask for {want} CPUs, Ray sees {have}. Give the job more CPUs.")
    reserve = cfg["memory"]["reserve_gb"]
    worst = sum(gb for _name, gb in plan_memory(cfg, pools, height, width))
    if worst > total_gb() - reserve:
        raise ConfigError(
            f"worst case {worst:.0f} GiB leaves less than reserve_gb {reserve} free; lower io.queue_frames or actor counts"
        )
