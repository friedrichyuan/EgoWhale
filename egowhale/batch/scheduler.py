"""The scheduler: keeps every GPU lane fed from CPU readers, and moves each episode's units through the DAG."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from egowhale.batch.actors import make_actors
from egowhale.batch.dag import POOL, STEPS, VISUAL, Job, take
from egowhale.batch.report import Report, boot, logger
from egowhale.batch.resources import INFLIGHT, available_gb, preflight
from egowhale.media import frame_size
from egowhale.step import Result, short_error
from egowhale.visual.inpaint import LABELS

# GPU batches without an OOM before a halved frame budget is doubled again.
RECOVER = 8
# Seconds the scheduler waits on Ray before it writes a TensorBoard point anyway.
WAIT = 5.0


@dataclass(eq=False)
class VisualLane:
    """One visual GPU actor, its CPU readers, and its writer."""

    stage: str
    label: str
    actor: Any
    readers: list
    writer: Any
    # ref -> (reader, frames) while decoding.
    busy: dict = field(default_factory=dict)
    # (ref, units, frames) once decoded.
    ready: list = field(default_factory=list)
    # ref -> frames on the GPU.
    inflight: dict = field(default_factory=dict)

    def load(self) -> int:
        return len(self.busy) + len(self.ready) + len(self.inflight)

    def held(self) -> int:
        return sum(frames for _reader, frames in self.busy.values()) + sum(frames for _ref, _units, frames in self.ready) + sum(
            self.inflight.values()
        )


@dataclass(eq=False)
class ActionLane:
    """One cuRobo actor with its reader and writer. While one stage computes, the next one's input is read."""

    actor: Any
    reader: Any
    writer: Any
    read: Any = None
    gpu: Any = None
    job: Job | None = None
    stage: str | None = None
    next_read: Any = None
    next_job: Job | None = None
    next_stage: str | None = None
    next_data: Any = None

    def idle(self) -> bool:
        return self.read is None and self.gpu is None and self.job is None and self.next_read is None and self.next_data is None

    def wants_next(self) -> bool:
        return self.gpu is not None and self.next_read is None and self.next_data is None


class Scheduler:
    def __init__(self, jobs: list[Job], cfg: dict, free: dict, report: Report):
        self.cfg = cfg
        self.report = report
        self.waiting = list(jobs)
        self.active: list[Job] = []
        # ref -> (handler, args): handler(ref, *args) runs when the ref is ready.
        self.pending: dict = {}
        self.free = free
        self.budget = {"inpaint": cfg["piece_frames"], "depth": cfg["nodes"]["depth"]["frames_per_forward"]}
        self.configured = dict(self.budget)
        # Batches in a row that ran without an OOM; after RECOVER of them a halved budget doubles back.
        self.calm = {stage: 0 for stage in self.budget}
        io = cfg["io"]
        self.queue_count = io["queue_per_lane"]
        self.queue_frames = io["queue_frames"]
        self.reserve = cfg["memory"]["reserve_gb"]
        readers = [actor for _label, actor in free.pop("feed")]
        writers = [actor for _label, actor in free.pop("write")]
        per = io["readers_per_lane"]
        self.lanes: list[VisualLane] = []
        for stage in VISUAL:
            for label, actor in free.pop(POOL[stage]):
                slot = len(self.lanes)
                self.lanes.append(VisualLane(stage, label, actor, readers[slot * per : (slot + 1) * per], writers[slot]))
        self.action_lanes = [
            ActionLane(actor, reader, writer)
            for (_label, actor), (_rl, reader), (_wl, writer) in zip(
                free.pop("action"), free.pop("action_read"), free.pop("action_write"), strict=True
            )
        ]

    # Bookkeeping ---------------------------------------------------------------------------------------------

    def apply(self, units, rows, stage: str) -> None:
        noted = False
        for (job, index), (ran, seconds, error) in zip(units, rows):
            if error:
                # Pieces already queued may still fail after the first error; the first one says why.
                if not job.failed:
                    where = f"{stage}" if index is None or len(job.spans) == 1 else f"{stage}[{index}]"
                    job.error = f"{where}: {error}"
                job.failed = True
            else:
                job.done.add((stage, index))
                if ran:
                    job.ran = True
                    self.report.note(stage, job.frames_of(index), seconds)
                    noted = True
            job.running.discard((stage, index))
            if job.finished() and job in self.active:
                self.report.record(job)
                self.active.remove(job)
        if noted:
            self.report.tick()

    # Visual lanes --------------------------------------------------------------------------------------------

    def fill_one(self, lane: VisualLane) -> bool:
        """Start decoding more pieces for this lane, if it has room in count, frames, and host memory."""
        if lane.load() >= self.queue_count:
            return False
        owners = [owner for owner, _frames in lane.busy.values()]
        reader = next((item for item in lane.readers if item not in owners), None)
        if reader is None:
            return False
        room = self.queue_frames - lane.held() if lane.load() else max(self.queue_frames, self.cfg["piece_frames"])
        if room <= 0:
            return False
        available = available_gb()
        if available is not None and available < self.reserve:
            return False
        chosen = take(self.active, lane.stage, self.budget.get(lane.stage, 0), room)
        if not chosen:
            return False
        for job, index in chosen:
            job.running.add((lane.stage, index))
        frames = sum(job.frames_of(index) for job, index in chosen)
        units = [(str(job.src), str(job.where(index)), *job.spans[index]) for job, index in chosen]
        ref = reader.prepare.remote(lane.stage, units)
        lane.busy[ref] = (reader, frames)
        self.pending[ref] = (self.on_feed, (lane, chosen))
        return True

    def fill(self) -> None:
        """Keep every lane's queue full. Scarce work goes to the least loaded lane first."""
        progress = True
        while progress:
            progress = False
            for lane in sorted(self.lanes, key=VisualLane.held):
                progress |= self.fill_one(lane)

    def launch(self, lane: VisualLane) -> None:
        while len(lane.inflight) < INFLIGHT and lane.ready:
            data, chosen, frames = lane.ready.pop(0)
            if all(job.failed for job, _index in chosen):
                # Every episode in this batch failed while it was decoding; its pieces need no GPU.
                for job, index in chosen:
                    job.running.discard((lane.stage, index))
                continue
            launched = lane.actor.consume.remote(data, self.budget.get(lane.stage))
            lane.inflight[launched] = frames
            self.pending[launched] = (self.on_visual, (lane, chosen))

    def on_feed(self, ref, lane: VisualLane, chosen) -> None:
        _reader, frames = lane.busy.pop(ref)
        lane.ready.append((ref, chosen, frames))
        self.launch(lane)
        self.fill()

    def on_visual(self, ref, lane: VisualLane, chosen) -> None:
        import ray

        chosen = list(chosen)
        stage = lane.stage
        lane.inflight.pop(ref, None)
        self.launch(lane)
        self.fill()
        try:
            payload = ray.get(ref)
        except Exception as exc:
            self.apply(chosen, [Result(False, 0.0, short_error(exc))] * len(chosen), stage)
            return
        rows = [Result(*row) for row in payload["results"]]
        if len(rows) != len(chosen):
            rows = [Result(False, 0.0, "batch result mismatch")] * len(chosen)
        elif stage in self.budget and "budget" in payload:
            self.adjust_budget(stage, int(payload["budget"]))
        save_ref = payload.get("save_ref")
        if save_ref is None:
            self.apply(chosen, rows, stage)
            return
        stored = lane.writer.commit.remote(save_ref)
        self.pending[stored] = (self.on_write, (stage, chosen, rows))

    def adjust_budget(self, stage: str, given: int) -> None:
        if given < self.budget[stage]:
            self.budget[stage], self.calm[stage] = given, 0
            return
        self.calm[stage] += 1
        if self.calm[stage] >= RECOVER and self.budget[stage] < self.configured[stage]:
            self.budget[stage], self.calm[stage] = min(self.configured[stage], self.budget[stage] * 2), 0

    def on_write(self, ref, stage: str, chosen, rows) -> None:
        import ray

        try:
            ray.get(ref)
        except Exception as exc:
            rows = [Result(False, 0.0, short_error(exc))] * len(chosen)
        self.apply(chosen, rows, stage)

    # Action lanes --------------------------------------------------------------------------------------------

    def feed_actions(self) -> None:
        taken = {id(job) for lane in self.action_lanes for job in (lane.job, lane.next_job) if job is not None}
        for lane in self.action_lanes:
            idle = lane.idle()
            if not idle and not lane.wants_next():
                continue
            chosen = next(
                ((job, stage) for job in self.active if id(job) not in taken for stage, _index in job.ready() if POOL[stage] == "action"),
                None,
            )
            if chosen is None:
                continue
            job, stage = chosen
            job.running.add((stage, None))
            taken.add(id(job))
            ref = lane.reader.load.remote(stage, str(job.src), str(job.dst))
            if idle:
                lane.read, lane.job, lane.stage = ref, job, stage
                self.pending[ref] = (self.on_action_read, (lane,))
            else:
                lane.next_read, lane.next_job, lane.next_stage = ref, job, stage
                self.pending[ref] = (self.on_action_next, (lane,))

    def on_action_read(self, ref, lane: ActionLane) -> None:
        lane.read = None
        launched = lane.actor.compute.remote(lane.stage, str(lane.job.dst), ref)
        lane.gpu = launched
        self.pending[launched] = (self.on_action, (lane,))

    def on_action_next(self, ref, lane: ActionLane) -> None:
        lane.next_read = None
        lane.next_data = ref
        if lane.gpu is None:
            self.promote(lane)

    def promote(self, lane: ActionLane) -> None:
        job, stage, data = lane.next_job, lane.next_stage, lane.next_data
        lane.next_job = lane.next_stage = lane.next_data = None
        lane.job, lane.stage = job, stage
        launched = lane.actor.compute.remote(stage, str(job.dst), data)
        lane.gpu = launched
        self.pending[launched] = (self.on_action, (lane,))

    def on_action(self, ref, lane: ActionLane) -> None:
        import ray

        job, stage = lane.job, lane.stage
        lane.gpu = lane.job = lane.stage = None
        if lane.next_data is not None:
            self.promote(lane)
        try:
            payload = ray.get(ref)
        except Exception as exc:
            self.apply([(job, None)], [Result(False, 0.0, short_error(exc))], stage)
            return
        if not payload["ok"]:
            self.apply([(job, None)], [Result(False, 0.0, payload["error"])], stage)
            return
        rows = [Result(True, payload["seconds"])]
        if payload["save_ref"] is None:
            self.apply([(job, None)], rows, stage)
            return
        stored = lane.writer.save.remote(payload["save_ref"])
        self.pending[stored] = (self.on_write, (stage, [(job, None)], rows))

    # CPU pools -----------------------------------------------------------------------------------------------

    def feed_runners(self) -> None:
        for job in self.active:
            for stage, index in job.ready():
                pool = POOL[stage]
                if stage in VISUAL or pool == "action" or not self.free[pool]:
                    continue
                label, actor = self.free[pool].pop()
                method = actor.run.remote(stage, str(job.src), str(job.dst), index)
                job.running.add((stage, index))
                self.pending[method] = (self.on_run, (job, stage, index, pool, label, actor))

    def on_run(self, ref, job: Job, stage: str, index, pool: str, label: str, actor) -> None:
        import ray

        try:
            rows = [Result(*ray.get(ref))]
        except Exception as exc:
            rows = [Result(False, 0.0, short_error(exc))]
        self.apply([(job, index)], rows, stage)
        self.free[pool].append((label, actor))

    # Loop ----------------------------------------------------------------------------------------------------

    def pump(self) -> None:
        while len(self.active) < self.cfg["memory"]["inflight"] and self.waiting:
            self.active.append(self.waiting.pop(0))
        self.fill()
        for lane in self.lanes:
            self.launch(lane)
        self.feed_actions()
        self.feed_runners()

    def run(self) -> None:
        import ray

        # A stall or an actor error ends the run; the board still gets its last point.
        try:
            while self.waiting or self.active or self.pending:
                self.pump()
                if not self.pending:
                    for job in list(self.active):
                        if job.finished():
                            self.report.record(job)
                            self.active.remove(job)
                    if not self.waiting and not self.pending:
                        break
                    if not self.active:
                        continue
                    available = available_gb()
                    if available is None or available >= self.reserve:
                        stuck = [f"{job.name} {[unit for unit in job.units() if unit not in job.done][:3]}" for job in self.active[:3]]
                        raise RuntimeError(f"scheduler stalled with nothing running: {stuck}")
                    # Host memory is below the reserve, so no piece may start. Wait for it to free up.
                    time.sleep(1.0)
                    continue
                ready, _rest = ray.wait(list(self.pending), num_returns=1, fetch_local=False, timeout=WAIT)
                if not ready:
                    self.report.tick()
                    continue
                handler, args = self.pending.pop(ready[0])
                handler(ready[0], *args)
        finally:
            self.report.close()


def serve(jobs: list[Job], cfg: dict, pools: dict[str, int], share: dict[str, float]) -> None:
    import ray

    memory = cfg["memory"]
    if not ray.is_initialized():
        store = int(memory["object_store_gb"] * 2**30) or None
        ray.init(ignore_reinit_error=True, log_to_driver=False, logging_level=logging.ERROR, object_store_memory=store)
    height, width = frame_size(jobs[0].src.with_suffix(".mp4"))
    preflight(cfg, pools, share, height, width)
    journal = logger()
    model = cfg["nodes"]["inpaint"]["model"]
    free = make_actors(pools, share, cfg["io"]["cpus"], cfg["nodes"]["depth"]["window"], model)
    boot(free, journal, pools, {"inpaint": LABELS[model]})
    # Render and curate draw with EGL on the action cards, which cuRobo leaves mostly idle.
    cards = sorted({device.split(":")[-1] for device in ray.get([actor.ready.remote() for _label, actor in free["action"]])})
    ray.get([actor.egl.remote(cards[index % len(cards)]) for pool in ("render", "cpu") for index, (_label, actor) in enumerate(free[pool])])
    out = Path(cfg["out"])
    split = [job for job in jobs if len(job.spans) > 1]
    journal.info(f"{len(jobs)} episodes, {len(split)} split into {sum(len(job.spans) for job in split)} pieces    out {out}")
    report = Report(out, {name: pools[POOL[name]] for name in STEPS})
    journal.info(f"tensorboard --logdir {report.board}")
    Scheduler(jobs, cfg, free, report).run()
