"""Pieces, grouping, error text, and the scheduler's dependency order. No GPU, no Ray."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from egowhale import paths, step
from egowhale import pieces as pieces_
from egowhale.batch.dag import DEPS, PIECED, POOL, STAGES, VISUAL, Job, take


def _episode(tmp_path, frames: int, video_frames: int | None = None):
    src = tmp_path / "task" / "0.hdf5"
    src.parent.mkdir(parents=True)
    with h5py.File(src, "w") as handle:
        handle["transforms/camera"] = np.repeat(np.eye(4)[None], frames, axis=0)
    writer = cv2.VideoWriter(str(src.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for _ in range(frames if video_frames is None else video_frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return src


@pytest.mark.parametrize("frames,limit", [(1, 600), (600, 600), (601, 600), (1200, 600), (1201, 600), (65, 40)])
def test_split_covers_every_frame_in_equal_pieces(frames, limit):
    spans = pieces_.split(frames, limit)
    assert spans[0][0] == 0 and spans[-1][1] == frames
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    sizes = [stop - start for start, stop in spans]
    assert max(sizes) <= limit and max(sizes) - min(sizes) <= 1


def test_plan_is_written_once_and_kept_for_the_same_limit(tmp_path):
    src = _episode(tmp_path, 65)
    dst = tmp_path / "out"
    assert pieces_.plan(src, dst, 40) == [(0, 32), (32, 65)]
    marked = json.loads((dst / paths.PIECES).read_text())
    assert marked == {"frames": 65, "limit": 40, "pieces": [[0, 32], [32, 65]]}
    assert pieces_.pieces(dst) == [(0, 32), (32, 65)]
    assert pieces_.plan(src, dst, 600) == [(0, 65)]


def test_plan_refuses_a_video_shorter_than_the_poses(tmp_path):
    src = _episode(tmp_path, 65, video_frames=60)
    with pytest.raises(ValueError, match="60 frames"):
        pieces_.plan(src, tmp_path / "out", 40)


def test_located_and_missing_follow_pieces(tmp_path):
    dst = tmp_path / "out"
    dst.mkdir()
    (dst / paths.PIECES).write_text(json.dumps({"frames": 10, "limit": 5, "pieces": [[0, 5], [5, 10]]}))
    assert pieces_.located(dst, paths.MASKS) == [dst / "pieces/00" / paths.MASKS, dst / "pieces/01" / paths.MASKS]
    assert pieces_.located(dst, paths.GRIPPER) == [dst / paths.GRIPPER]
    first = dst / "pieces/00" / paths.MASKS
    first.parent.mkdir(parents=True)
    first.touch()
    assert pieces_.missing(dst, (paths.MASKS,)) == [paths.MASKS]
    (dst / "pieces/01" / paths.MASKS).parent.mkdir(parents=True)
    (dst / "pieces/01" / paths.MASKS).touch()
    assert pieces_.missing(dst, (paths.MASKS,)) == []


def test_short_error_never_fails():
    assert step.short_error(RuntimeError("first\nlast")) == "last"
    assert step.short_error(AssertionError()) == "AssertionError"
    assert step.short_error(ValueError("  \n ")) == "ValueError"


def test_groups_within_keeps_oversized_items_alone():
    assert step.groups_within([0, 1, 2, 3], [10, 10, 10, 10], 25) == [[0, 1], [2, 3]]
    assert step.groups_within([0, 1, 2], [50, 10, 10], 25) == [[0], [1, 2]]
    assert step.groups_within([], [], 25) == []


def test_run_grouped_halves_on_oom_and_reports_other_errors():
    tried = []

    def forward(group):
        tried.append(list(group))
        if len(group) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate")
        if 5 in group:
            raise AssertionError()

    errors, budget = step.run_grouped(list(range(8)), {index: 1 for index in range(8)}, 8, forward)
    assert tried[0] == list(range(8))
    assert sorted(index for group in tried if len(group) <= 2 for index in group) == list(range(8))
    # 8, then 4 and 4 out of memory: every OOM halves the budget, as before the shared helper.
    assert budget == 1
    assert errors == {4: "AssertionError", 5: "AssertionError"}


def test_run_grouped_gives_up_on_a_single_item_oom():
    def forward(group):
        raise RuntimeError("CUDA out of memory")

    errors, budget = step.run_grouped([0], {0: 3}, 3, forward)
    assert errors == {0: "CUDA out of memory"} and budget == 3


def simulate(count: int, pools: dict[str, int], inflight: int, pieces: int = 1, fail=None) -> list[tuple]:
    """Single-threaded stand-in for the scheduler: each pool has N slots and finishes the oldest unit first."""
    fail = fail or {}
    spans = [(index * 100, (index + 1) * 100) for index in range(pieces)]
    jobs = [Job(Path(f"task/{index}.hdf5"), Path(f"out/{index}"), 100, spans=spans) for index in range(count)]
    for job in jobs:
        job.done.clear()
    free = dict(pools)
    waiting = list(jobs)
    active: list[Job] = []
    running: list[tuple] = []
    trace = []
    for _ in range(100000):
        if not (waiting or active or running):
            return trace
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
            if all(job.finished() for job in active):
                return trace
            raise RuntimeError("no ready stage and nothing running")
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
    raise RuntimeError("scheduler did not finish")


POOLS = {"action": 1, "segment": 1, "inpaint": 1, "depth": 1, "render": 1, "cpu": 2}


def _order_respects_deps(trace, pieces):
    ended = {}
    for position, (episode, stage, index, event) in enumerate(trace):
        if event == "end":
            ended[(episode, stage, index)] = position
    for position, (episode, stage, index, event) in enumerate(trace):
        if event != "start":
            continue
        for dep in DEPS[stage]:
            if dep in PIECED:
                wanted = [index] if index is not None else list(range(pieces))
            else:
                wanted = [None]
            for piece in wanted:
                assert ended.get((episode, dep, piece), len(trace)) < position, (episode, stage, index, dep, piece)


@pytest.mark.parametrize("pieces", [1, 3])
def test_simulated_schedule_finishes_every_unit_in_dependency_order(pieces):
    trace = simulate(4, POOLS, inflight=2, pieces=pieces)
    _order_respects_deps(trace, pieces)
    units = {(stage, index) for stage in DEPS for index in (range(pieces) if stage in PIECED else (None,))}
    for episode in range(4):
        assert {(s, i) for e, s, i, event in trace if e == episode and event == "end"} == units


def test_simulated_failure_stops_only_that_episode():
    trace = simulate(3, POOLS, inflight=3, pieces=2, fail={(1, "inpaint"): "boom"})
    assert not [row for row in trace if row[0] == 1 and row[1] in ("composite", "curate")]
    assert [row for row in trace if row[0] == 0 and row[1] == "curate" and row[3] == "end"]
    assert [row for row in trace if row[0] == 2 and row[1] == "curate" and row[3] == "end"]


def test_every_stage_has_a_pool():
    assert set(DEPS) == set(POOL)


def test_stage_table_derives_the_old_constants():
    assert list(DEPS) == ["retarget", "segment", "base_ik", "inpaint", "approach", "depth", "render", "composite", "curate"]
    assert PIECED == ("segment", "inpaint", "depth", "render")
    assert VISUAL == ("segment", "inpaint", "depth")
    assert {name for name, stage in STAGES.items() if stage.pool == "action"} == {"retarget", "base_ik", "approach"}


def test_take_batches_depth_only():
    jobs = [Job(Path(f"t/{index}.hdf5"), Path(f"o/{index}"), 100, spans=[(0, 40), (40, 80)]) for index in range(3)]
    for job in jobs:
        job.done.clear()
    assert [(job.src.stem, index) for job, index in take(jobs, "segment", 0, 1000)] == [("0", 0)]
    assert len(take(jobs, "depth", 120, 1000)) == 3
    assert len(take(jobs, "depth", 1000, 100)) == 2
    assert take(jobs, "inpaint", 0, 1000) == []
