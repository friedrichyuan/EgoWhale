"""What a batch run tells its operator: the boot table, one JSON line per episode, and TensorBoard."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from egowhale.batch.dag import STEPS, Job
from egowhale.batch.resources import available_gb, total_gb
from egowhale.step import short_error

# (pool, display name, category, model or role)
PLACED = (
    ("segment", "Segment", "Visual", "SAM 3 + Cutie"),
    ("inpaint", "Inpaint", "Visual", "STTN"),
    ("depth", "Depth", "Visual", "DA3-GIANT"),
    ("action", "Action", "Action", "cuRobo"),
    ("render", "Render", "Action", "MuJoCo"),
    ("cpu", "Composite", "CPU", "paste"),
    ("cpu", "Curate", "CPU", "checks"),
    ("feed", "Feed", "IO", "prefetch"),
    ("write", "Write", "IO", "store"),
    ("action_read", "Action-Read", "IO", "prefetch"),
    ("action_write", "Action-Write", "IO", "store"),
)
# Episodes whose inpainted scene and composite go to TensorBoard, as a share of the run.
WATCHED = 0.1
# Frames of one watched episode, and their height in the side-by-side video.
COMPARE_FRAMES = 32
COMPARE_HEIGHT = 360


def logger():
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> │ <level>{level:<7}</level> │ {message}", colorize=True)
    return logger


def _row(log, status: str, category: str, node: str, count: str, devices: str, model: str) -> None:
    line = f"{status:<10} │ {category:<8} │ {node:<18} │ {count:>5} │ {devices:<10} │ {model}"
    if status == "Running":
        log.success(line)
    else:
        log.info(line)


def boot(free: dict, log, pools: dict[str, int], models: dict[str, str] | None = None) -> None:
    """Print each pool as it starts and once all its actors answer. `models` replaces PLACED's model column."""
    import ray

    models = models or {}
    placed = [(pool, name, category, models.get(pool, model)) for pool, name, category, model in PLACED]
    _row(log, "Status", "Category", "Node", "Count", "Device", "Model")
    seen = set()
    for pool, name, category, model in placed:
        if pool in seen or pool not in free:
            continue
        seen.add(pool)
        _row(log, "Starting", category, name, str(pools.get(pool, "")), "", model)
    pending = [(pool, actor.ready.remote()) for pool, slots in free.items() for _label, actor in slots]
    left = {pool: len(slots) for pool, slots in free.items()}
    found: dict[str, list[str]] = {pool: [] for pool in free}
    while pending:
        done, _rest = ray.wait([ref for _pool, ref in pending], num_returns=1)
        pool, ref = next(item for item in pending if item[1] is done[0])
        pending.remove((pool, ref))
        found[pool].append(ray.get(ref))
        left[pool] -= 1
        if left[pool] == 0:
            _pool, name, category, model = next((entry for entry in placed if entry[0] == pool), (pool, pool, "", ""))
            # Compact device list, e.g. "0, 1, 2" or "cpu".
            devices = ", ".join(dict.fromkeys(found[pool])).replace("cuda:", "")
            _row(log, "Running", category, name, str(pools.get(pool, "")), devices, model)


def watched(count: int) -> set[int]:
    """Evenly spaced episode indexes, WATCHED of them."""
    if count <= 0:
        return set()
    keep = min(count, max(1, int(round(count * WATCHED))))
    return {int(index) for index in np.unique(np.round(np.linspace(0, count - 1, keep)).astype(int))}


class Meter:
    """TensorBoard scalars, at most one point per MIN_INTERVAL seconds, plus one at each finished episode.

    system/ok and system/failed count episodes; system/fps is cumulative frames over elapsed seconds and
    system/fps_5m the same over the last five minutes; system/memory_used_gb is host RAM when written.
    node/<stage>/fps uses the pool's cumulative compute time, times its actor count, so it shows what the pool
    can do: two episodes on one pool of N actors show N times the per-actor rate.
    The step is wall-clock seconds since boot.
    """

    MIN_INTERVAL = 30
    REPORTED = frozenset({"segment", "inpaint", "depth", "base_ik", "render", "composite"})
    WINDOW = 300

    def __init__(self, actors: dict[str, int]):
        self.actors = actors
        self.wall0 = time.perf_counter()
        self.ok = 0
        self.fail = 0
        self.frames = 0
        self.nodes = {name: {"frames": 0, "seconds": 0.0} for name in STEPS}
        # (wall time, cumulative frames) at each finished episode, for fps_5m.
        self._window: list[tuple[float, int]] = []
        self._last_write = -self.MIN_INTERVAL

    def note(self, stage: str, frames: int, seconds: float) -> None:
        node = self.nodes[stage]
        node["frames"] += frames
        node["seconds"] += seconds

    def finish(self, job: Job) -> None:
        if job.failed:
            self.fail += 1
        else:
            self.ok += 1
            if job.ran:
                self.frames += job.frames
                self._window.append((time.perf_counter(), self.frames))

    def write(self, writer, force: bool = False) -> None:
        """Write a point. Skips silently if called too soon unless `force` is True."""
        now = time.perf_counter()
        if not force and now - self._last_write < self.MIN_INTERVAL:
            return
        self._last_write = now
        step = max(0, int(now - self.wall0))
        writer.add_scalar("system/ok", self.ok, step)
        writer.add_scalar("system/failed", self.fail, step)
        elapsed = now - self.wall0
        if self.frames and elapsed >= 1:
            writer.add_scalar("system/fps", self.frames / elapsed, step)
        self._window = [(t, f) for t, f in self._window if t >= now - self.WINDOW]
        if len(self._window) >= 2:
            dt = self._window[-1][0] - self._window[0][0]
            df = self._window[-1][1] - self._window[0][1]
            if dt >= 1:
                writer.add_scalar("system/fps_5m", df / dt, step)
        available = available_gb()
        if available is not None:
            writer.add_scalar("system/memory_used_gb", total_gb() - available, step)
        for name, node in self.nodes.items():
            if name in self.REPORTED and node["seconds"] >= 0.05:
                writer.add_scalar(f"node/{name}/fps", node["frames"] / node["seconds"] * self.actors[name], step)


class Report:
    """batch.jsonl and TensorBoard for one run."""

    def __init__(self, out: Path, actors: dict[str, int]):
        from torch.utils.tensorboard import SummaryWriter

        self.log = out / "batch.jsonl"
        self.board = out / "tb"
        self.writer = SummaryWriter(log_dir=str(self.board))
        self.meter = Meter(actors)

    def note(self, stage: str, frames: int, seconds: float) -> None:
        self.meter.note(stage, frames, seconds)

    def tick(self) -> None:
        self.meter.write(self.writer)

    def record(self, job: Job) -> None:
        row = {
            "episode": job.name,
            "ok": not job.failed,
            "error": job.error,
            "frames": job.frames,
            "pieces": len(job.spans),
            "seconds": round(time.perf_counter() - job.started, 2),
        }
        self.meter.finish(job)
        self.meter.write(self.writer, force=True)
        if row["ok"] and job.watch:
            try:
                _compare(job, self.writer)
            except Exception as exc:
                row["compare"] = short_error(exc)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def close(self) -> None:
        self.meter.write(self.writer, force=True)
        self.writer.flush()
        self.writer.close()


def _compare(job: Job, writer) -> None:
    """Up to COMPARE_FRAMES frames of the inpainted scene beside the composite. Only those frames are decoded."""
    from egowhale.media import read_frames, video_shape
    from egowhale.paths import COMPOSITE, INPAINT, PREFIX

    prefix = len(np.load(job.dst / PREFIX)["qpos"])
    count = min(job.frames, video_shape(job.dst / COMPOSITE)[0] - max(prefix - 1, 0))
    if count <= 0:
        return
    pick = np.unique(np.round(np.linspace(0, count - 1, min(COMPARE_FRAMES, count))).astype(int))
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
    writer.add_video("compare/" + job.name.replace("/", "_"), video, job.index, fps=4)
    writer.flush()


def _fit(frame: np.ndarray) -> np.ndarray:
    import cv2

    height, width = frame.shape[:2]
    if height <= COMPARE_HEIGHT:
        return np.ascontiguousarray(frame)
    width = int(width * COMPARE_HEIGHT / height) // 2 * 2
    return cv2.resize(frame, (width, COMPARE_HEIGHT), interpolation=cv2.INTER_AREA)


def _pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    import cv2

    if left.shape != right.shape:
        right = cv2.resize(right, (left.shape[1], left.shape[0]))
    return np.ascontiguousarray(np.concatenate([left, right], axis=1))
