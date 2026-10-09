"""Ray pools for the episode DAG. One heavy model stays on each GPU.

Episodes are split into pieces when they are first read. Segment, inpaint, depth, and render run per piece;
the action stages run per episode; composite merges the pieces into one video and curate runs on that.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

# moviepy 1.0.3 predates Python 3.12 and warns on its own regex literals.
warnings.filterwarnings("ignore", category=SyntaxWarning, module=r"moviepy(\.|$)")

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from egowhale import config as configs
from egowhale.action.approach import Approach
from egowhale.action.base_ik import BaseIK
from egowhale.action.curate import Curate
from egowhale.action.retarget import Retarget
from egowhale.step import missing, piece_dir, plan
from egowhale.visual.composite import Composite
from egowhale.visual.depth import Depth
from egowhale.visual.inpaint import Inpaint
from egowhale.visual.render import Render
from egowhale.visual.segment import Segment

# Later stages wait on these. A per-piece stage waits on the same piece of another per-piece stage;
# a per-episode stage waits on every piece. Only inpaint waits inside the visual branch; everything meets at composite.
DEPS = {
    "retarget": (),
    "segment": (),
    "base_ik": ("retarget",),
    "inpaint": ("segment",),
    "approach": ("base_ik",),
    "depth": (),
    "render": ("retarget", "base_ik", "approach"),
    "composite": ("segment", "inpaint", "depth", "approach", "render"),
    "curate": ("composite",),
}
PIECED = ("segment", "inpaint", "depth", "render")
POOL = {
    "retarget": "action",
    "base_ik": "action",
    "approach": "action",
    "segment": "segment",
    "inpaint": "inpaint",
    "depth": "depth",
    "render": "render",
    "composite": "cpu",
    "curate": "cpu",
}
VISUAL = ("segment", "inpaint", "depth")
STEPS = {
    "retarget": Retarget,
    "segment": Segment,
    "inpaint": Inpaint,
    "depth": Depth,
    "base_ik": BaseIK,
    "approach": Approach,
    "render": Render,
    "composite": Composite,
    "curate": Curate,
}


class Job:
    """One episode. Work is tracked as units: (stage, piece index) for per-piece stages, (stage, None) otherwise."""

    def __init__(self, src: Path, dst: Path, limit: int, spans: list[tuple[int, int]] | None = None):
        self.src = src
        self.dst = dst
        self.spans = spans if spans is not None else plan(src, dst, limit)
        self.frames = self.spans[-1][1]
        self.done = {unit for unit in self.units() if _unit_done(self, unit)}
        self.running: set[tuple[str, int | None]] = set()
        self.failed = False
        self.error = ""
        self.ran = False
        self.started = time.perf_counter()

    def units(self) -> list[tuple[str, int | None]]:
        return [(stage, index) for stage in DEPS for index in (range(len(self.spans)) if stage in PIECED else (None,))]

    def ready(self) -> list[tuple[str, int | None]]:
        if self.failed:
            return []
        return [unit for unit in self.units() if unit not in self.done and unit not in self.running and self._met(unit)]

    def _met(self, unit) -> bool:
        stage, index = unit
        for dep in DEPS[stage]:
            if dep not in PIECED:
                wanted = [(dep, None)]
            elif index is not None:
                wanted = [(dep, index)]
            else:
                wanted = [(dep, piece) for piece in range(len(self.spans))]
            if any(item not in self.done for item in wanted):
                return False
        return True

    def frames_of(self, index: int | None) -> int:
        if index is None:
            return self.frames
        start, stop = self.spans[index]
        return stop - start

    def where(self, index: int | None) -> Path:
        return self.dst if index is None else piece_dir(self.dst, index, len(self.spans))

    def finished(self) -> bool:
        return self.failed or len(self.done) == len(self.units())


def _unit_done(job: Job, unit) -> bool:
    stage, index = unit
    return all((job.where(index) / name).is_file() for name in STEPS[stage].makes)


def execute(step, src, dst, index=None) -> tuple[bool, float]:
    """Run one stage, or one piece of it. Return whether it computed, and how long that took. Skips are (False, 0)."""
    os.environ["EGOWHALE_QUIET"] = "1"
    src, dst = Path(src), Path(dst)
    if index is None:
        outputs = [dst / name for name in step.makes]
    else:
        from egowhale.step import pieces

        outputs = [piece_dir(dst, index, len(pieces(dst))) / name for name in step.makes]
    if all(path.is_file() for path in outputs):
        return False, 0.0
    absent = missing(dst, step.needs)
    if absent:
        raise FileNotFoundError(f"{step.name} missing {absent}")
    buffer = io.StringIO()
    started = time.perf_counter()
    with contextlib.redirect_stdout(buffer):
        if index is None:
            step.run(src, dst)
        else:
            step.run_piece(src, dst, index)
    if not all(path.is_file() for path in outputs):
        raise FileNotFoundError(f"{step.name} did not write {[str(path) for path in outputs if not path.is_file()]}")
    return True, time.perf_counter() - started


def _device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "cpu"
    return "cuda:" + visible.split(",")[0]


def _egl(device: str) -> None:
    """MuJoCo EGL ignores CUDA_VISIBLE_DEVICES and would render on physical GPU 0."""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = device


def _actors(pools: dict[str, int], share: dict[str, float], cpus: dict[str, int], window: int):
    """Every actor lives for the whole run. `share` is the GPU fraction for pools that stack on one card."""
    import ray

    gpu = ray.remote(num_gpus=1, max_concurrency=1)
    # The next piece waits on the compute lock while this one copies its result out.
    pipe = ray.remote(num_gpus=1, max_concurrency=2)
    cpu = ray.remote(num_cpus=cpus["cpu"], max_concurrency=1)

    @pipe
    class SegmentActor:
        def __init__(self):
            from egowhale.visual.segment import load_tracker

            self.step = Segment()
            self.step._processor, self.step._cutie = load_tracker()

        def ready(self):
            return _device()

        def consume(self, payload):
            return self.step.consume(payload)

    @pipe
    class InpaintActor:
        def __init__(self):
            self.step = Inpaint()
            self.step.load()

        def ready(self):
            return _device()

        def consume(self, payload, budget):
            return self.step.consume(payload, budget)

    @pipe
    class DepthActor:
        def __init__(self):
            from egowhale.visual.depth import load_model

            self.step = Depth()
            self.step.window = window
            self.step._model = load_model()

        def ready(self):
            return _device()

        def consume(self, payload, budget):
            return self.step.consume(payload, budget)

    @gpu
    class ActionActor:
        def __init__(self):
            self.steps = {"retarget": Retarget(), "base_ik": BaseIK(), "approach": Approach()}

        def ready(self):
            return _device()

        def compute(self, stage, dst, data):
            import ray

            started = time.perf_counter()
            try:
                out = self.steps[stage].compute(data)
            except Exception as exc:
                return {"ok": False, "error": str(exc).splitlines()[-1], "seconds": 0.0, "save_ref": None}
            packed = {"stage": stage, "dst": dst, "out": out}
            return {"ok": True, "seconds": time.perf_counter() - started, "save_ref": ray.put(packed)}

    @cpu
    class CpuActor:
        def __init__(self):
            os.environ.pop("EGOWHALE_VLM_API_KEY", None)
            self.steps = {"composite": Composite(), "curate": Curate()}

        def ready(self):
            return "cpu"

        def egl(self, device):
            _egl(device)

        def run(self, stage, src, dst, index=None):
            return execute(self.steps[stage], src, dst, index)

    @ray.remote(num_cpus=cpus["render"], max_concurrency=1)
    class RenderActor:
        def __init__(self):
            self.step = Render()

        def ready(self):
            return "cpu"

        def egl(self, device):
            _egl(device)

        def run(self, stage, src, dst, index=None):
            return execute(self.step, src, dst, index)

    @ray.remote(num_cpus=cpus["feed"], max_concurrency=8)
    class FeedActor:
        def ready(self):
            return "cpu"

        def prepare(self, stage, units):
            from egowhale.visual.feed import prepare

            return prepare(stage, units)

    @ray.remote(num_cpus=cpus["write"], max_concurrency=1)
    class StoreActor:
        def ready(self):
            return "cpu"

        def commit(self, saves):
            from egowhale.visual.store import commit

            commit(saves)
            return True

    @ray.remote(num_cpus=cpus["action_read"], max_concurrency=1)
    class ActionReadActor:
        def ready(self):
            return "cpu"

        def load(self, stage, src, dst):
            src, dst = Path(src), Path(dst)
            if stage == "retarget":
                from egowhale.action.retarget import read_episode

                return read_episode(src)
            if stage == "base_ik":
                from egowhale.action.base_ik import read_gripper

                return read_gripper(dst)
            if stage == "approach":
                from egowhale.action.approach import read_ik

                return read_ik(dst)
            raise ValueError(stage)

    @ray.remote(num_cpus=cpus["action_write"], max_concurrency=1)
    class ActionWriteActor:
        def ready(self):
            return "cpu"

        def save(self, packed):
            dst = Path(packed["dst"])
            stage = packed["stage"]
            out = packed["out"]
            if stage == "retarget":
                from egowhale.action.retarget import save_gripper

                save_gripper(dst, out)
            elif stage == "base_ik":
                from egowhale.action.base_ik import save_base

                save_base(dst, out)
            elif stage == "approach":
                from egowhale.action.approach import save_prefix

                save_prefix(dst, out)
            else:
                raise ValueError(stage)
            return True

    classes = {
        "segment": SegmentActor,
        "inpaint": InpaintActor,
        "depth": DepthActor,
        "action": ActionActor,
        "cpu": CpuActor,
        "render": RenderActor,
        "feed": FeedActor,
        "write": StoreActor,
        "action_read": ActionReadActor,
        "action_write": ActionWriteActor,
    }
    made = {}

    def make(names) -> None:
        for name in names:
            cls = classes[name].options(num_gpus=share[name]) if name in share else classes[name]
            made[name] = [(f"{name}{index}", cls.remote()) for index in range(pools[name])]

    # Whole cards are placed before fractional ones so a shared card cannot take a slot a whole actor needs.
    # Among fractional pools, a ridden node goes first so each of its actors takes a card of its own.
    make([name for name in classes if name not in share])
    ray.get([actor.ready.remote() for name in ("inpaint", "depth") if name not in share for _label, actor in made.get(name, [])])
    for name in share:
        make([name])
        ray.get([actor.ready.remote() for _label, actor in made[name]])
    return {name: made[name] for name in classes}


_PLACED = (
    # (pool, display name, category, model/role)
    ("segment",      "Segment",      "Visual",  "SAM 3 + Cutie"),
    ("inpaint",      "Inpaint",      "Visual",  "ProPainter"),
    ("depth",        "Depth",        "Visual",  "DA3-GIANT"),
    ("action",       "Action",       "Action",  "cuRobo"),
    ("render",       "Render",       "Action",  "MuJoCo"),
    ("cpu",          "Composite",    "CPU",     "paste"),
    ("cpu",          "Curate",       "CPU",     "checks"),
    ("feed",         "Feed",         "IO",      "prefetch"),
    ("write",        "Write",        "IO",      "store"),
    ("action_read",  "Action-Read",  "IO",      "prefetch"),
    ("action_write", "Action-Write", "IO",      "store"),
)


def _logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> │ <level>{level:<7}</level> │ {message}", colorize=True)
    return logger


def _row(log, status: str, category: str, node: str, count: str, device: str, model: str) -> None:
    # Device: keep only the numeric part(s) of cuda:N to stay narrow, e.g. "0,1,2" or "cpu".
    short = device.replace("cuda:", "").replace("cpu", "cpu") if device else ""
    line = f"{status:<10} │ {category:<8} │ {node:<18} │ {count:>5} │ {short:<10} │ {model}"
    if status == "Running":
        log.success(line)
    else:
        log.info(line)


def _boot(free: dict, log, pools: dict[str, int]) -> None:
    import ray

    _row(log, "Status", "Category", "Node", "Count", "Device", "Model")
    seen = set()
    for pool, name, category, model in _PLACED:
        if pool in seen or pool not in free:
            continue
        seen.add(pool)
        _row(log, "Starting", category, name, str(pools.get(pool, "")), "", model)
    pending = []
    left = {pool: len(slots) for pool, slots in free.items()}
    found: dict[str, list[str]] = {pool: [] for pool in free}
    for pool, slots in free.items():
        for _label, actor in slots:
            pending.append((pool, actor.ready.remote()))
    while pending:
        done, _rest = ray.wait([item[1] for item in pending], num_returns=1)
        ref = done[0]
        pool, handle = next(item for item in pending if item[1] is ref)
        pending.remove((pool, ref))
        found[pool].append(ray.get(handle))
        left[pool] -= 1
        if left[pool] == 0:
            entry = next((e for e in _PLACED if e[0] == pool), None)
            name = entry[1] if entry else pool
            category = entry[2] if entry else ""
            model = entry[3] if entry else ""
            # Compact device list: strip "cuda:" prefix and deduplicate.
            raw = ", ".join(dict.fromkeys(found[pool]))
            devices = raw.replace("cuda:", "")
            _row(log, "Running", category, name, str(pools.get(pool, "")), devices, model)


def _take(active: list[Job], stage: str, budget: int, room: int) -> list[tuple[Job, int]]:
    """Pieces, oldest episode first. Depth adds more pieces until one forward's worth of frames is queued.

    `room` is the frames the lane may still take; the first piece is always taken by an empty lane.
    """
    chosen: list[tuple[Job, int]] = []
    total = 0
    for job in active:
        for unit_stage, index in job.ready():
            if unit_stage != stage:
                continue
            frames = job.frames_of(index)
            if chosen and (stage != "depth" or total >= budget or total + frames > room):
                return chosen
            if not chosen and frames > room:
                return chosen
            chosen.append((job, index))
            total += frames
    return chosen


def _apply(units, rows, stage, meter, writer, log, active) -> None:
    noted = False
    for (job, index), row in zip(units, rows):
        ran, seconds = row[0], row[1]
        error = row[2] if len(row) > 2 else ""
        if error:
            job.failed = True
            where = f"{stage}" if index is None or len(job.spans) == 1 else f"{stage}[{index}]"
            job.error = f"{where}: {error}"
        else:
            job.done.add((stage, index))
            if ran:
                job.ran = True
                meter.note(stage, job.frames_of(index), seconds)
                noted = True
            if index is None or all((stage, piece) in job.done for piece in range(len(job.spans))):
                meter.episode(stage)
        job.running.discard((stage, index))
        if job.finished() and job in active:
            _record(job, log, writer, meter)
            active.remove(job)
    if noted:
        meter.write(writer)


def _available_gb() -> float | None:
    """MemAvailable from /proc/meminfo; None where it does not exist."""
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        return None
    return None


def _total_gb() -> float:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def _plan_memory(cfg: dict, pools: dict[str, int], height: int, width: int) -> list[tuple[str, float]]:
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
        # Crop frames, crop masks, the crop's original, and the half-size background.
        "inpaint": half * 3 * 3 + half,
        # DA3 input at 336 on the long side, float32.
        "depth": 3 * 4 * 336 * 336 * height // width,
    }
    readers = cfg["io"]["readers_per_lane"]
    lanes = {stage: pools[stage] for stage in VISUAL}
    rows = []
    for stage in VISUAL:
        decoding = readers * piece * (frame + held[stage])
        rows.append((f"{stage} queues", lanes[stage] * (queue * held[stage] + decoding) / 2**30))
    # Results waiting for the writer: masks are bool at full size, the painted crop plus background, depth float32 at half.
    saved = {"segment": height * width, "inpaint": half * 3 * 2, "depth": half * 4}
    rows.append(("visual results", sum(lanes[stage] * 2 * piece * saved[stage] for stage in VISUAL) / 2**30))
    # Render holds rgb, part, gripper mask, and depth at half size.
    rows.append(("render", pools["render"] * piece * half * (3 + 1 + 1 + 4) / 2**30))
    # Composite holds a piece of background, render, scene depth, robot depth, masks at full and half size, and the output.
    rows.append(("composite", pools["cpu"] * piece * (half * (3 + 3 + 1 + 1 + 4 + 4 + 1) + height * width) / 2**30))
    # Python, CUDA, and model weights in every worker process.
    workers = sum(pools[name] for name in ("segment", "inpaint", "depth", "action")) * 4 + sum(
        pools[name] for name in ("render", "cpu", "feed", "write", "action_read", "action_write")
    ) * 0.5
    rows.append(("worker processes", workers))
    return rows


def serve(jobs: list[Job], cfg: dict, pools: dict[str, int], share: dict[str, float]) -> None:
    import ray
    from torch.utils.tensorboard import SummaryWriter

    out = Path(cfg["out"])
    log = out / "batch.jsonl"
    board = out / "tb"
    memory = cfg["memory"]
    if not ray.is_initialized():
        store = int(memory["object_store_gb"] * 2**30) or None
        ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR, object_store_memory=store)
    have = int(ray.cluster_resources().get("GPU", 0))
    need = round(sum(pools[name] * share.get(name, 1.0) for name in ("segment", "inpaint", "depth", "action")))
    if need > have:
        raise SystemExit(f"GPU pools ask for {need} devices, cluster has {have}")
    cpus = cfg["io"]["cpus"]
    # GPU actors need one free CPU while they are being created.
    want = sum(pools[name] * count for name, count in cpus.items()) + 1
    have = int(ray.cluster_resources().get("CPU", 0))
    if want > have:
        raise SystemExit(f"CPU pools ask for {want} CPUs, Ray sees {have}. Give the job more CPUs.")
    journal = _logger()
    from egowhale.media import video_shape

    _count, height, width, _fps = video_shape(jobs[0].src.with_suffix(".mp4"))
    worst = sum(gb for _name, gb in _plan_memory(cfg, pools, height, width))
    if worst > _total_gb() - memory["reserve_gb"]:
        raise SystemExit(f"worst case {worst:.0f} GiB leaves less than reserve_gb {memory['reserve_gb']} free; lower io.queue_frames or actor counts")
    free = _actors(pools, share, cpus, cfg["nodes"]["depth"]["window"])
    _boot(free, journal, pools)
    # Render and curate draw with EGL on the action cards, which cuRobo leaves mostly idle.
    cards = sorted({device.split(":")[-1] for device in ray.get([actor.ready.remote() for _label, actor in free["action"]])})
    ray.get([actor.egl.remote(cards[index % len(cards)]) for pool in ("render", "cpu") for index, (_label, actor) in enumerate(free[pool])])
    split = sum(len(job.spans) > 1 for job in jobs)
    journal.info(f"{len(jobs)} episodes, {split} split into {sum(len(job.spans) for job in jobs if len(job.spans) > 1)} pieces    out {out}")
    journal.info(f"tensorboard --logdir {board}")
    writer = SummaryWriter(log_dir=str(board))
    meter = _Meter({name: pools[POOL[name]] for name in STEPS})
    waiting = list(jobs)
    active: list[Job] = []
    pending = {}
    budget = {"inpaint": cfg["piece_frames"], "depth": cfg["nodes"]["depth"]["frames_per_forward"]}
    readers = [actor for _label, actor in free.pop("feed")]
    writers = [actor for _label, actor in free.pop("write")]
    per = cfg["io"]["readers_per_lane"]
    queue_count = cfg["io"]["queue_per_lane"]
    queue_frames = cfg["io"]["queue_frames"]
    reserve = memory["reserve_gb"]
    lanes = []
    slot = 0
    for stage in VISUAL:
        for label, actor in free.pop(POOL[stage]):
            lanes.append(
                {
                    "stage": stage,
                    "label": label,
                    "actor": actor,
                    "readers": readers[slot * per : (slot + 1) * per],
                    "writer": writers[slot],
                    # ref -> (reader, frames) while decoding; (ref, units, frames) once decoded; ref -> frames on the GPU.
                    "busy": {},
                    "ready": [],
                    "inflight": {},
                }
            )
            slot += 1
    action_lanes = []
    for (_label, actor), (_rl, reader), (_wl, store) in zip(free.pop("action"), free.pop("action_read"), free.pop("action_write")):
        action_lanes.append(
            {
                "actor": actor,
                "reader": reader,
                "writer": store,
                "read": None,
                "gpu": None,
                "job": None,
                "stage": None,
                "next_read": None,
                "next_job": None,
                "next_stage": None,
                "next_data": None,
            }
        )

    def consume(lane, data, chosen, frames) -> None:
        if lane["stage"] == "segment":
            launched = lane["actor"].consume.remote(data)
        else:
            launched = lane["actor"].consume.remote(data, budget[lane["stage"]])
        lane["inflight"][launched] = frames
        pending[launched] = ("gpu", lane, chosen)

    def load(lane) -> int:
        return len(lane["busy"]) + len(lane["ready"]) + len(lane["inflight"])

    def held(lane) -> int:
        return (
            sum(frames for _reader, frames in lane["busy"].values())
            + sum(frames for _ref, _units, frames in lane["ready"])
            + sum(lane["inflight"].values())
        )

    def fill_one(lane) -> bool:
        """Start decoding more pieces for this lane, if it has room in count, frames, and host memory."""
        if load(lane) >= queue_count:
            return False
        reader = next((item for item in lane["readers"] if item not in [owner for owner, _frames in lane["busy"].values()]), None)
        if reader is None:
            return False
        stage = lane["stage"]
        room = queue_frames - held(lane) if load(lane) else max(queue_frames, cfg["piece_frames"])
        if room <= 0:
            return False
        available = _available_gb()
        if available is not None and available < reserve:
            meter.paused += 1
            return False
        chosen = _take(active, stage, budget.get(stage, 0), room)
        if not chosen:
            return False
        for job, index in chosen:
            job.running.add((stage, index))
        frames = sum(job.frames_of(index) for job, index in chosen)
        units = [(str(job.src), str(job.where(index)), *job.spans[index]) for job, index in chosen]
        ref = reader.prepare.remote(stage, units)
        lane["busy"][ref] = (reader, frames)
        pending[ref] = ("feed", lane, chosen)
        return True

    def fill() -> None:
        """Keep every lane's queue full. Scarce work goes to the least loaded lane first."""
        progress = True
        while progress:
            progress = False
            for lane in sorted(lanes, key=held):
                progress |= fill_one(lane)

    def launch(lane) -> None:
        while len(lane["inflight"]) < 2 and lane["ready"]:
            data, chosen, frames = lane["ready"].pop(0)
            consume(lane, data, chosen, frames)

    def promote_action(lane) -> None:
        job = lane["next_job"]
        stage = lane["next_stage"]
        data = lane["next_data"]
        lane["next_job"] = None
        lane["next_stage"] = None
        lane["next_data"] = None
        lane["job"] = job
        lane["stage"] = stage
        launched = lane["actor"].compute.remote(stage, str(job.dst), data)
        lane["gpu"] = launched
        pending[launched] = ("action", lane)

    def pump() -> None:
        while len(active) < memory["inflight"] and waiting:
            active.append(waiting.pop(0))
        fill()
        for lane in lanes:
            launch(lane)
        taken = set()
        for lane in action_lanes:
            if lane["job"] is not None:
                taken.add(id(lane["job"]))
            if lane["next_job"] is not None:
                taken.add(id(lane["next_job"]))
        for lane in action_lanes:
            idle = lane["read"] is None and lane["gpu"] is None and lane["job"] is None and lane["next_read"] is None and lane["next_data"] is None
            want_next = lane["gpu"] is not None and lane["next_read"] is None and lane["next_data"] is None
            if not idle and not want_next:
                continue
            chosen = None
            for job in active:
                if id(job) in taken:
                    continue
                for stage, _index in job.ready():
                    if POOL[stage] != "action":
                        continue
                    chosen = (job, stage)
                    break
                if chosen:
                    break
            if chosen is None:
                continue
            job, stage = chosen
            job.running.add((stage, None))
            taken.add(id(job))
            ref = lane["reader"].load.remote(stage, str(job.src), str(job.dst))
            if idle:
                lane["read"] = ref
                lane["job"] = job
                lane["stage"] = stage
                pending[ref] = ("aread", lane)
            else:
                lane["next_read"] = ref
                lane["next_job"] = job
                lane["next_stage"] = stage
                pending[ref] = ("anext", lane)
        for job in active:
            for stage, index in job.ready():
                pool = POOL[stage]
                if stage in VISUAL or pool == "action":
                    continue
                if not free[pool]:
                    continue
                label, actor = free[pool].pop()
                method = actor.run.remote(stage, str(job.src), str(job.dst), index)
                job.running.add((stage, index))
                pending[method] = ("run", job, stage, index, pool, label, actor)

    while waiting or active or pending:
        pump()
        if not pending:
            for job in list(active):
                if job.finished():
                    _record(job, log, writer, meter)
                    active.remove(job)
            if not waiting and not pending:
                break
            if not active:
                continue
            available = _available_gb()
            if available is None or available >= reserve:
                stuck = [f"{_episode(job)} {[unit for unit in job.units() if unit not in job.done][:3]}" for job in active[:3]]
                raise RuntimeError(f"scheduler stalled with nothing running: {stuck}")
            # Host memory is below the reserve, so no piece may start. Wait for it to free up.
            time.sleep(1.0)
            continue
        ready_refs, _rest = ray.wait(list(pending), num_returns=1, fetch_local=False, timeout=5.0)
        if not ready_refs:
            meter.write(writer)
            continue
        ref = ready_refs[0]
        kind = pending.pop(ref)
        if kind[0] == "feed":
            lane, chosen = kind[1], kind[2]
            _reader, frames = lane["busy"].pop(ref)
            lane["ready"].append((ref, chosen, frames))
            launch(lane)
            fill()
            continue
        if kind[0] == "aread":
            lane = kind[1]
            lane["read"] = None
            launched = lane["actor"].compute.remote(lane["stage"], str(lane["job"].dst), ref)
            lane["gpu"] = launched
            pending[launched] = ("action", lane)
            continue
        if kind[0] == "anext":
            lane = kind[1]
            lane["next_read"] = None
            lane["next_data"] = ref
            if lane["gpu"] is None:
                promote_action(lane)
            continue
        if kind[0] == "run":
            _job, stage, index, pool, label, actor = kind[1:]
            try:
                ran, seconds = ray.get(ref)
                rows = [(ran, seconds, "")]
            except Exception:
                message = traceback.format_exc().strip().splitlines()[-1]
                rows = [(False, 0.0, message)]
            _apply([(_job, index)], rows, stage, meter, writer, log, active)
            free[pool].append((label, actor))
            continue
        if kind[0] in ("write", "awrite"):
            stage, chosen, rows = kind[1:]
            try:
                ray.get(ref)
            except Exception:
                message = traceback.format_exc().strip().splitlines()[-1]
                rows = [(False, 0.0, message)] * len(chosen)
            _apply(chosen, rows, stage, meter, writer, log, active)
            continue
        if kind[0] == "action":
            lane = kind[1]
            job = lane["job"]
            stage = lane["stage"]
            lane["gpu"] = None
            lane["job"] = None
            lane["stage"] = None
            if lane["next_data"] is not None:
                promote_action(lane)
            try:
                payload = ray.get(ref)
            except Exception:
                message = traceback.format_exc().strip().splitlines()[-1]
                _apply([(job, None)], [(False, 0.0, message)], stage, meter, writer, log, active)
                continue
            if not payload["ok"]:
                _apply([(job, None)], [(False, 0.0, payload["error"])], stage, meter, writer, log, active)
                continue
            rows = [(True, payload["seconds"], "")]
            if payload["save_ref"] is None:
                _apply([(job, None)], rows, stage, meter, writer, log, active)
                continue
            stored = lane["writer"].save.remote(payload["save_ref"])
            pending[stored] = ("awrite", stage, [(job, None)], rows)
            continue
        lane, chosen = kind[1], list(kind[2])
        stage = lane["stage"]
        lane["inflight"].pop(ref, None)
        launch(lane)
        fill()
        try:
            payload = ray.get(ref)
        except Exception:
            message = traceback.format_exc().strip().splitlines()[-1]
            _apply(chosen, [(False, 0.0, message)] * len(chosen), stage, meter, writer, log, active)
            continue
        rows = payload["results"]
        if len(rows) != len(chosen):
            rows = [(False, 0.0, "batch result mismatch")] * len(chosen)
        elif stage in budget and "budget" in payload:
            budget[stage] = min(budget[stage], int(payload["budget"]))
        save_ref = payload.get("save_ref")
        if save_ref is None:
            _apply(chosen, rows, stage, meter, writer, log, active)
            continue
        stored = lane["writer"].commit.remote(save_ref)
        pending[stored] = ("write", stage, chosen, rows)
    writer.flush()
    writer.close()
    elapsed = max(time.perf_counter() - meter.wall0, 1e-6)
    journal.info(f"done  {meter.ok} ok  {meter.fail} fail  {meter.frames / elapsed:.2f} fps  peak used {meter.peak_gb:.0f} GiB")
    ray.shutdown()


def _episode(job: Job) -> str:
    return f"{job.src.parent.name}/{job.src.stem}"


def _watch(count: int) -> set[int]:
    if count <= 0:
        return set()
    keep = min(count, max(1, int(round(count * 0.1))))
    indexes = np.unique(np.round(np.linspace(0, count - 1, keep)).astype(int))
    return set(int(index) for index in indexes)


class _Meter:
    """system/* counts this run only; frames are counted once per finished episode. node/*/fps is the whole pool's
    speed while busy. Skips add episodes, not frames.

    Stages that share a pool (retarget, base_ik, approach) each report as if the pool ran only that stage.
    """

    def __init__(self, actors: dict[str, int]):
        self.actors = actors
        self.wall0 = time.perf_counter()
        self.episodes = 0
        self.ok = 0
        self.fail = 0
        self.frames = 0
        self.paused = 0
        self.peak_gb = 0.0
        self.nodes = {name: {"frames": 0, "seconds": 0.0, "episodes": 0} for name in STEPS}
        self._last = -1

    def note(self, stage: str, frames: int, seconds: float) -> None:
        node = self.nodes[stage]
        node["frames"] += frames
        node["seconds"] += seconds

    def episode(self, stage: str) -> None:
        self.nodes[stage]["episodes"] += 1

    def finish(self, job: Job) -> None:
        self.episodes += 1
        if job.failed:
            self.fail += 1
            return
        self.ok += 1
        if job.ran:
            self.frames += job.frames

    def write(self, writer) -> None:
        step = self._step()
        writer.add_scalar("system/episodes", self.episodes, step)
        writer.add_scalar("system/frames", self.frames, step)
        elapsed = time.perf_counter() - self.wall0
        if self.frames and elapsed >= 1:
            writer.add_scalar("system/fps", self.frames / elapsed, step)
        available = _available_gb()
        if available is not None:
            used = _total_gb() - available
            self.peak_gb = max(self.peak_gb, used)
            writer.add_scalar("system/memory_used_gb", used, step)
        writer.add_scalar("system/decode_paused", self.paused, step)
        for name, node in self.nodes.items():
            if node["seconds"] >= 0.05:
                writer.add_scalar(f"node/{name}/fps", node["frames"] / node["seconds"] * self.actors[name], step)

    def _step(self) -> int:
        """TensorBoard step is wall seconds since boot. Milliseconds made 100s look like 100000."""
        step = int(time.perf_counter() - self.wall0)
        if step <= self._last:
            step = self._last + 1
        self._last = step
        return step


def _record(job: Job, log: Path, writer, meter: _Meter) -> None:
    row = {
        "episode": _episode(job),
        "ok": not job.failed,
        "error": job.error,
        "frames": job.frames,
        "pieces": len(job.spans),
        "seconds": round(time.perf_counter() - job.started, 2),
    }
    meter.finish(job)
    meter.write(writer)
    if row["ok"] and job.watch:
        try:
            _compare(job, writer)
        except Exception as exc:
            row["compare"] = str(exc).splitlines()[-1]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _compare(job: Job, writer) -> None:
    """Up to 32 frames of the inpainted scene beside the composite. Only those frames are decoded."""
    from egowhale.media import read_frames, video_shape
    from egowhale.step import COMPOSITE, INPAINT, PREFIX

    prefix = len(np.load(job.dst / PREFIX)["qpos"])
    count = min(job.frames, video_shape(job.dst / COMPOSITE)[0] - max(prefix - 1, 0))
    if count <= 0:
        return
    pick = np.unique(np.round(np.linspace(0, count - 1, min(32, count))).astype(int))
    after = read_frames(job.dst / COMPOSITE, pick + max(prefix - 1, 0))
    before = []
    for index, (start, stop) in enumerate(job.spans):
        local = pick[(pick >= start) & (pick < stop)] - start
        if len(local):
            before.append(read_frames(job.where(index) / INPAINT, local))
    before = np.concatenate(before)
    count = min(len(before), len(after))
    frames = [_pair(_fit(before[index]), _fit(after[index])) for index in range(count)]
    video = np.stack(frames).transpose(0, 3, 1, 2)[None]
    writer.add_video("compare/" + _episode(job).replace("/", "_"), video, job.index, fps=4)
    writer.flush()


def _fit(frame: np.ndarray) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    if height <= 360:
        return np.ascontiguousarray(frame)
    width = int(width * 360 / height) // 2 * 2
    return cv2.resize(frame, (width, 360), interpolation=cv2.INTER_AREA)


def _pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    import cv2

    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return np.ascontiguousarray(np.concatenate([left, right], axis=1))


def simulate(count: int, pools: dict[str, int], inflight: int, pieces: int = 1, fail: dict[tuple[int, str], str] | None = None) -> list[tuple]:
    """Single-threaded stand-in: each pool has N slots and finishes the oldest unit first."""
    fail = fail or {}
    spans = [(index * 100, (index + 1) * 100) for index in range(pieces)]
    jobs = [Job(Path(f"task/{index}.hdf5"), Path(f"out/{index}"), 100, spans=spans) for index in range(count)]
    for job in jobs:
        job.done.clear()
    free = {name: size for name, size in pools.items()}
    waiting = list(jobs)
    active: list[Job] = []
    running: list[tuple] = []
    trace = []
    guard = 0
    while waiting or active or running:
        guard += 1
        if guard > 100000:
            raise RuntimeError("scheduler did not finish")
        while len(active) < inflight and waiting:
            active.append(waiting.pop(0))
        for job in active:
            number = int(job.src.stem)
            for stage, index in job.ready():
                pool = POOL[stage]
                if free[pool] <= 0:
                    continue
                free[pool] -= 1
                job.running.add((stage, index))
                running.append((number, stage, index, pool))
                trace.append((number, stage, index, "start"))
        if not running:
            for job in list(active):
                if job.finished():
                    active.remove(job)
            if not running and all(job.finished() for job in active):
                break
            if not running:
                raise RuntimeError("no ready stage and nothing running")
            continue
        number, stage, index, pool = running.pop(0)
        job = jobs[number]
        job.running.discard((stage, index))
        free[pool] += 1
        reason = fail.get((number, stage))
        if reason:
            job.failed = True
            job.error = reason
            trace.append((number, stage, index, "fail"))
        else:
            job.done.add((stage, index))
            trace.append((number, stage, index, "end"))
        if job.finished() and job in active:
            active.remove(job)
    return trace


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the episode DAG on Ray pools, as a config in cfgs/ describes.")
    parser.add_argument("config", type=Path, help="YAML config, e.g. cfgs/bench8_len1000.yaml")
    parser.add_argument("--name", help="experiment name; results go to <out>/<name>/")
    parser.add_argument("--limit", type=int, help="run only the first N episodes")
    args = parser.parse_args()
    cfg = configs.load(args.config, name=args.name, limit=args.limit)
    sources = configs.episodes(cfg)
    if not sources:
        raise SystemExit("no episodes")
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    import yaml

    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    jobs = [Job(src, out / src.parent.name / src.stem, cfg["piece_frames"]) for src in sources]
    pools, share = pools_from(cfg)
    watch = _watch(len(jobs))
    for index, job in enumerate(jobs):
        job.index = index
        job.watch = index in watch
    serve(jobs, cfg, pools, share)


if __name__ == "__main__":
    main()
