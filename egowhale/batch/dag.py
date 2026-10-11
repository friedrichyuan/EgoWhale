"""The episode DAG: each stage's class, pool, and dependencies, and one episode's units of work.

Episodes are split into pieces when they are first read. Segment, inpaint, depth, and render run per piece;
the action stages run per episode; composite merges the pieces into one video and curate runs on that.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

from egowhale.action.approach import Approach
from egowhale.action.base_ik import BaseIK
from egowhale.action.retarget import Retarget
from egowhale.pieces import piece_dir, plan
from egowhale.quality.curate import Curate
from egowhale.step import Step
from egowhale.visual.composite import Composite
from egowhale.visual.depth import Depth
from egowhale.visual.inpaint import Inpaint
from egowhale.visual.render import Render
from egowhale.visual.segment import Segment


class Stage(NamedTuple):
    step: type[Step]
    # The actor pool that runs it. Visual GPU pools are named after their stage.
    pool: str
    # Later stages wait on these. A per-piece stage waits on the same piece of another per-piece stage;
    # a per-episode stage waits on every piece.
    deps: tuple[str, ...] = ()


# In this order, ready units of one episode are handed out. Only inpaint waits inside the visual branch;
# everything meets at composite.
STAGES = {
    "retarget": Stage(Retarget, "action"),
    "segment": Stage(Segment, "segment"),
    "base_ik": Stage(BaseIK, "action", ("retarget",)),
    "inpaint": Stage(Inpaint, "inpaint", ("segment",)),
    "approach": Stage(Approach, "action", ("base_ik",)),
    "depth": Stage(Depth, "depth"),
    "render": Stage(Render, "render", ("retarget", "base_ik", "approach")),
    "composite": Stage(Composite, "cpu", ("segment", "inpaint", "depth", "approach", "render")),
    "curate": Stage(Curate, "cpu", ("composite",)),
}
DEPS = {name: stage.deps for name, stage in STAGES.items()}
POOL = {name: stage.pool for name, stage in STAGES.items()}
STEPS = {name: stage.step for name, stage in STAGES.items()}
PIECED = tuple(name for name, stage in STAGES.items() if stage.step.per_piece)
# Stages fed by CPU readers and computed in batches on their own GPU lanes.
VISUAL = tuple(name for name, stage in STAGES.items() if hasattr(stage.step, "consume"))
# Stages read, computed, and saved one episode at a time on the action lanes.
ACTION = tuple(name for name, stage in STAGES.items() if stage.pool == "action")
# Pools whose actors hold GPU memory.
GPU_POOLS = ("segment", "inpaint", "depth", "action")


class Job:
    """One episode. Work is tracked as units: (stage, piece index) for per-piece stages, (stage, None) otherwise."""

    def __init__(self, src: Path, dst: Path, limit: int, spans: list[tuple[int, int]] | None = None):
        self.src = src
        self.dst = dst
        self.spans = spans if spans is not None else plan(src, dst, limit)
        self.frames = self.spans[-1][1]
        self.done = {unit for unit in self.units() if unit_done(self, unit)}
        self.running: set[tuple[str, int | None]] = set()
        self.failed = False
        self.error = ""
        self.ran = False
        self.started = time.perf_counter()
        # Position in the run, and whether its composite goes to TensorBoard.
        self.index = 0
        self.watch = False

    @property
    def name(self) -> str:
        return f"{self.src.parent.name}/{self.src.stem}"

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


def unit_done(job: Job, unit) -> bool:
    stage, index = unit
    return all((job.where(index) / name).is_file() for name in STEPS[stage].makes)


def take(active: list[Job], stage: str, budget: int, room: int) -> list[tuple[Job, int]]:
    """Pieces, oldest episode first. Depth adds more pieces until one forward's worth of frames is queued.

    `room` is the frames the lane may still take; the first piece is always taken by an empty lane.
    Segment and inpaint take one piece at a time: a batch of them would not run faster.
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
