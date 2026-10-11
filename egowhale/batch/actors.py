"""Ray actors. Every actor lives for the whole run; each GPU actor holds one model."""

from __future__ import annotations

import os
import time
from pathlib import Path

from egowhale.batch.dag import ACTION, STEPS, VISUAL
from egowhale.step import Result, execute, short_error


def device() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return "cpu"
    return "cuda:" + visible.split(",")[0]


def egl(card: str) -> None:
    """MuJoCo EGL ignores CUDA_VISIBLE_DEVICES and would render on physical GPU 0."""
    os.environ["MUJOCO_EGL_DEVICE_ID"] = card


def _visual_step(name: str, inpaint_model: str, window: int):
    """A visual stage built with its options. Building one loads nothing."""
    options = {"inpaint": (inpaint_model,), "depth": (window,)}
    return STEPS[name](*options.get(name, ()))


def _visual_steps(inpaint_model: str, window: int) -> dict:
    return {name: _visual_step(name, inpaint_model, window) for name in VISUAL}


def _action_steps() -> dict:
    return {name: STEPS[name]() for name in ACTION}


class _Cpu:
    def ready(self) -> str:
        return "cpu"


class _Gpu:
    def ready(self) -> str:
        return device()


class _Visual(_Gpu):
    """One visual stage's model. `consume` computes one prefetched batch under the step's lock."""

    stage = ""

    def __init__(self, inpaint_model: str, window: int):
        self.step = _visual_step(self.stage, inpaint_model, window)
        self.step.load()

    def consume(self, payload, budget):
        return self.step.consume(payload, budget)


class SegmentActor(_Visual):
    stage = "segment"


class InpaintActor(_Visual):
    stage = "inpaint"


class DepthActor(_Visual):
    stage = "depth"


class ActionActor(_Gpu):
    def __init__(self):
        self.steps = _action_steps()

    def compute(self, stage, dst, data):
        import gc

        import ray
        import torch

        started = time.perf_counter()
        try:
            out = self.steps[stage].compute(data)
        except Exception as exc:
            return {"ok": False, "error": short_error(exc), "seconds": 0.0, "save_ref": None}
        finally:
            # cuRobo solvers sit in reference cycles; without this their GPU memory piles up episode after episode.
            gc.collect()
            torch.cuda.empty_cache()
        packed = {"stage": stage, "dst": dst, "out": out}
        return {"ok": True, "seconds": time.perf_counter() - started, "save_ref": ray.put(packed)}


class _Runner(_Cpu):
    """Runs whole CPU stages, or one piece of them, and draws with EGL on the card it is given."""

    stages: tuple[str, ...] = ()

    def __init__(self):
        self.steps = {name: STEPS[name]() for name in self.stages}

    def egl(self, card):
        egl(card)

    def run(self, stage, src, dst, index=None) -> Result:
        return execute(self.steps[stage], src, dst, index)


class CpuActor(_Runner):
    stages = ("composite", "curate")

    def __init__(self):
        # Batch runs do not call the VLM.
        os.environ.pop("EGOWHALE_VLM_API_KEY", None)
        super().__init__()


class RenderActor(_Runner):
    stages = ("render",)


class FeedActor(_Cpu):
    """Builds visual inputs on the CPU while the GPU is busy. Holds no weights."""

    def __init__(self, inpaint_model: str, window: int):
        self.steps = _visual_steps(inpaint_model, window)

    def prepare(self, stage, units):
        """`units` are (episode, piece dir, start, stop). A unit that cannot be read fails alone."""
        items = []
        for src, where, start, stop in units:
            try:
                item = self.steps[stage].prepare(Path(src), Path(where), int(start), int(stop))
                item["ok"] = True
            except Exception as exc:
                item = {"ok": False, "error": short_error(exc)}
            items.append(item)
        return {"items": items}


class StoreActor(_Cpu):
    """Writes visual results so the GPU actor can take its next batch."""

    def __init__(self, inpaint_model: str, window: int):
        self.steps = _visual_steps(inpaint_model, window)

    def commit(self, saves):
        for save in saves:
            self.steps[save["stage"]].store(save)
        return True


class ActionReadActor(_Cpu):
    def __init__(self):
        self.steps = _action_steps()

    def load(self, stage, src, dst):
        return self.steps[stage].read(Path(src), Path(dst))


class ActionWriteActor(_Cpu):
    def __init__(self):
        self.steps = _action_steps()

    def save(self, packed):
        self.steps[packed["stage"]].save(Path(packed["dst"]), packed["out"])
        return True


# pool -> (actor class, Ray options as a function of io.cpus, whether it takes (inpaint_model, window)).
# Visual GPU actors take two calls at once: the next batch waits on the step's lock while this one copies out.
POOLS = {
    "segment": (SegmentActor, lambda cpus: {"num_gpus": 1, "max_concurrency": 2}, True),
    "inpaint": (InpaintActor, lambda cpus: {"num_gpus": 1, "max_concurrency": 2}, True),
    "depth": (DepthActor, lambda cpus: {"num_gpus": 1, "max_concurrency": 2}, True),
    "action": (ActionActor, lambda cpus: {"num_gpus": 1, "max_concurrency": 1}, False),
    "cpu": (CpuActor, lambda cpus: {"num_cpus": cpus["cpu"], "max_concurrency": 1}, False),
    "render": (RenderActor, lambda cpus: {"num_cpus": cpus["render"], "max_concurrency": 1}, False),
    "feed": (FeedActor, lambda cpus: {"num_cpus": cpus["feed"], "max_concurrency": 8}, True),
    "write": (StoreActor, lambda cpus: {"num_cpus": cpus["write"], "max_concurrency": 1}, True),
    "action_read": (ActionReadActor, lambda cpus: {"num_cpus": cpus["action_read"], "max_concurrency": 1}, False),
    "action_write": (ActionWriteActor, lambda cpus: {"num_cpus": cpus["action_write"], "max_concurrency": 1}, False),
}


def make_actors(pools: dict[str, int], share: dict[str, float], cpus: dict[str, int], window: int, inpaint_model: str):
    """Every pool's actors as (label, handle). `share` is the GPU fraction for pools that stack on one card."""
    import ray

    made = {}

    def make(names) -> None:
        for name in names:
            cls, options, configured = POOLS[name]
            remote = ray.remote(**options(cpus))(cls)
            if name in share:
                remote = remote.options(num_gpus=share[name])
            args = (inpaint_model, window) if configured else ()
            made[name] = [(f"{name}{index}", remote.remote(*args)) for index in range(pools[name])]

    # Whole cards are placed before fractional ones so a shared card cannot take a slot a whole actor needs.
    # Among fractional pools, a ridden node goes first so each of its actors takes a card of its own.
    make([name for name in POOLS if name not in share])
    ray.get([actor.ready.remote() for name in ("inpaint", "depth") if name not in share for _label, actor in made.get(name, [])])
    for name in share:
        make([name])
        ray.get([actor.ready.remote() for _label, actor in made[name]])
    return {name: made[name] for name in POOLS}
