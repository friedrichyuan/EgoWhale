"""Benchmark one node with persistent workers and explicit input roots."""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from multiprocessing.util import Finalize
from pathlib import Path
from queue import Empty

from egowhale.config import DEFAULTS, ConfigError, episodes, ik_options, read_config
from egowhale.paths import IK, PREFIX
from egowhale.pieces import located, missing, pieces, plan


def prepare(src, dst, inputs, step, piece_frames):
    """Link only prerequisites; writes always belong to the fresh benchmark directory."""
    from egowhale.__main__ import STEPS

    roots = [] if inputs is None else ([inputs] if isinstance(inputs, Path) else list(inputs))
    plans = [(root / "pieces.json") for root in roots if (root / "pieces.json").is_file()]
    if plans and any(json.loads(p.read_text()) != json.loads(plans[0].read_text()) for p in plans[1:]):
        raise ValueError(f"Incompatible piece plans: {plans}")
    dst.mkdir(parents=True, exist_ok=True)
    if plans:
        shutil.copyfile(plans[0], dst / "pieces.json")
    else:
        plan(src, dst, piece_frames)
    if step.needs:
        if not roots:
            raise FileNotFoundError(f"{step.name} needs {step.needs}; provide --inputs-root")
        names = set(step.needs)
        for cls in STEPS:
            names.update(cls.makes)
        for name in sorted(names - set(step.makes)):
            for target in located(dst, name):
                sources = { (root / target.relative_to(dst)).resolve()
                            for root in roots if (root / target.relative_to(dst)).is_file() }
                if len(sources) > 1:
                    raise ValueError(f"Conflicting upstream artifact {target.relative_to(dst)}: {sources}")
                if sources:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(sources.pop())
        absent = missing(dst, step.needs)
        if absent:
            raise FileNotFoundError(f"Missing upstream files under {roots}: {absent}")


def processed_frames(step, dst, source_frames):
    from egowhale.robots.motion import frame_count, prefix_length, read_motion

    if step.name == "approach":
        return frame_count(read_motion(dst / PREFIX)), "output_prefix_frames"
    if step.name == "composite":
        from egowhale.media import video_shape
        return video_shape(dst / "visual/composite.mp4")[0], "output_video_frames"
    if step.name == "render":
        return source_frames + prefix_length(dst) - 1, "output_video_frames"
    if step.name == "base_ik":
        return frame_count(read_motion(dst / IK)), "episode_frames"
    return source_frames, "episode_frames"


def measure(step, src, dst):
    started = time.perf_counter()
    timing = {}
    # Action nodes already expose this boundary in their real single-node implementation.
    if all(callable(getattr(step, name, None)) for name in ("read", "compute", "save")):
        data = step.read(src, dst)
        read_at = time.perf_counter()
        result = step.compute(data)
        compute_at = time.perf_counter()
        step.save(dst, result)
        done = time.perf_counter()
        timing = dict(read_s=read_at-started, compute_s=compute_at-read_at, save_s=done-compute_at)
    else:
        step.run(src, dst)
        done = time.perf_counter()
    absent = missing(dst, step.makes)
    if absent:
        raise RuntimeError(f"{step.name} did not write {absent}")
    return dict(node_s=done-started, **timing)


def summarize(rows, load_s, wall_s):
    good = [r for r in rows if r["status"] == "ok"]
    steady = [r for r in good if not r["warmup"]]
    def aggregate(items):
        frames = sum(r["frames"] for r in items)
        seconds = sum(r["node_s"] for r in items)
        compute = sum(r.get("compute_s", 0.) for r in items)
        return dict(episodes=len(items), frames=frames, node_s=seconds,
                    node_fps=frames/seconds if seconds else None,
                    compute_fps=frames/compute if compute and all("compute_s" in r for r in items) else None)
    return dict(ok=len(good), failed=len(rows)-len(good), load_s=load_s, wall_s=wall_s,
                all_successful=aggregate(good), after_warmup=aggregate(steady),
                overall_fps=sum(r["frames"] for r in good)/wall_s if wall_s else None)


class _EpisodeRunner:
    """Reuse one node and its models for the episodes handled by this process."""

    def __init__(self, step, inputs_root, warmup, piece_frames):
        self.step = step
        self.inputs_root = inputs_root
        self.warmup = warmup
        self.piece_frames = piece_frames
        self.loaded = False
        self.successes = 0

    def stream(self, jobs):
        """Read and write episodes around the same sequential solver used by the action actors."""
        step = self.step
        stream = step.open_stream()
        source = iter(jobs)
        live = {}
        exhausted = False
        try:
            while not exhausted or live:
                arrivals = []
                for _ in range(stream.capacity - len(live)):
                    job = next(source, None)
                    if job is None:
                        exhausted = True
                        break
                    src, dst, key = job
                    row = dict(episode=str(src), output=str(dst), stage=step.name, status="failed")
                    started = time.perf_counter()
                    try:
                        inputs = self.inputs_root
                        roots = [root / key for root in inputs] if isinstance(inputs, list) else inputs / key if inputs else None
                        prepare(src, dst, roots, step, self.piece_frames)
                        read_at = time.perf_counter()
                        data = step.read(src, dst)
                        row["read_s"] = time.perf_counter() - read_at
                    except Exception as exc:
                        row.update(error=f"{type(exc).__name__}: {exc}", episode_wall_s=time.perf_counter() - started)
                        yield row, 0.0
                        continue
                    token = str(dst)
                    row["compute_s"] = 0.0
                    live[token] = (row, started)
                    arrivals.append((token, data))
                if not live:
                    continue
                before = time.perf_counter()
                completed = stream.advance(arrivals)
                share = (time.perf_counter() - before) / len(live)
                for row, _ in live.values():
                    row["compute_s"] += share
                for result in completed:
                    token = result["token"]
                    row, started = live.pop(token)
                    try:
                        if not result["ok"]:
                            raise RuntimeError(result["error"])
                        dst = Path(token)
                        save_at = time.perf_counter()
                        step.save(dst, result["out"])
                        row["save_s"] = time.perf_counter() - save_at
                        if absent := missing(dst, step.makes):
                            raise RuntimeError(f"{step.name} did not write {absent}")
                        row["frames"], row["frame_basis"] = processed_frames(step, dst, 0)
                        row["node_s"] = row["read_s"] + row["compute_s"] + row["save_s"]
                        row.update(status="ok", warmup=self.successes < self.warmup,
                                   fps=row["frames"] / max(row["node_s"], 1e-12))
                        self.successes += 1
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                    row["episode_wall_s"] = time.perf_counter() - started
                    yield row, 0.0
        finally:
            stream.close()

    def run(self, src, dst, key):
        step = self.step
        inputs = self.inputs_root
        row = dict(episode=str(src), output=str(dst), stage=step.name, status="failed")
        started = time.perf_counter()
        load_s = 0.0
        try:
            if isinstance(inputs, list):
                roots = [root / key for root in inputs]
            else:
                roots = inputs / key if inputs else None
            prepare(src, dst, roots, step, self.piece_frames)
            frames = pieces(dst)[-1][1]
            if not self.loaded:
                before = time.perf_counter()
                step.load()
                load_s = time.perf_counter() - before
                self.loaded = True
            row.update(measure(step, src, dst))
            row["frames"], row["frame_basis"] = processed_frames(step, dst, frames)
            row.update(status="ok", warmup=self.successes < self.warmup)
            self.successes += 1
            row["fps"] = row["frames"] / max(row["node_s"], 1e-12)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        if callable(getattr(step, "compute", None)):
            # cuRobo solver reference cycles must not accumulate between episodes.
            gc.collect()
        row["episode_wall_s"] = time.perf_counter() - started
        return row, load_s


def _jobs(sources, out):
    keys = [Path(src.parent.name) / src.stem for src in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("Episode task/name collisions or duplicates in input list")
    out.mkdir(parents=True, exist_ok=False)
    return [(src, out / key, key) for src, key in zip(sources, keys, strict=True)]


def _record(results, out, started):
    """Only the coordinator writes shared benchmark files."""
    rows = []
    load_s = 0.0

    def snapshot():
        summary = summarize(rows, load_s, time.perf_counter() - started)
        temp = out / ".summary.json.tmp"
        temp.write_text(json.dumps(summary, indent=2))
        temp.replace(out / "summary.json")
        return summary

    with (out / "results.jsonl").open("a") as handle:
        try:
            for row, loaded in results:
                rows.append(row)
                load_s += loaded
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                snapshot()
                print(json.dumps(row), flush=True)
        finally:
            summary = snapshot()
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def benchmark(step, sources, out, inputs_root=None, warmup=1, piece_frames=600):
    jobs = _jobs(sources, out)
    runner = _EpisodeRunner(step, inputs_root, warmup, piece_frames)
    def results():
        try:
            if getattr(step, "ik_method", None) == "base_ik_seq":
                yield from runner.stream(jobs)
            else:
                for job in jobs:
                    yield runner.run(*job)
        finally:
            step.release()

    return _record(results(), out, time.perf_counter())


def _bind_device(stage, device):
    os.environ["CUDA_VISIBLE_DEVICES"] = device if device is not None else "-1"
    if stage in ("render", "curate"):
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MUJOCO_EGL_DEVICE_ID"] = device


def _make_step(args):
    from egowhale.__main__ import STEPS
    from egowhale.robots.factory import make_step
    from egowhale.visual.depth import Depth
    from egowhale.visual.inpaint import Inpaint

    cls = next(s for s in STEPS if s.name == args.stage)
    if cls is Inpaint:
        return Inpaint(args.inpaint_model)
    if cls is Depth:
        return Depth(args.window)
    return make_step(cls, args.robot, ik_method=args.ik_method, batch_size=args.batch_size)


_worker_runner = None


def _init_worker(slots, args, inputs):
    global _worker_runner
    _bind_device(args.stage, slots.get())
    step = _make_step(args)
    _worker_runner = _EpisodeRunner(step, inputs, args.warmup, args.piece_frames)
    Finalize(None, step.release, exitpriority=10)


def _worker_run(job):
    row, load_s = _worker_runner.run(*job)
    row.update(worker_pid=os.getpid(), gpu=os.environ["CUDA_VISIBLE_DEVICES"])
    return row, load_s


def _worker_stream(device, args, inputs, jobs, results):
    _bind_device(args.stage, device)
    step = _make_step(args)
    runner = _EpisodeRunner(step, inputs, args.warmup, args.piece_frames)
    try:
        for row, load_s in runner.stream(jobs):
            row.update(worker_pid=os.getpid(), gpu=device)
            results.put((row, load_s))
    finally:
        step.release()


def _parallel_stream(jobs, devices, args, inputs):
    # Each device owns one fixed shard and one stream; publish results as episodes finish.
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager, ExitStack() as stack:
        results = manager.Queue()
        futures = []
        for index, device in enumerate(devices):
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx))
            futures.append(pool.submit(_worker_stream, device, args, inputs, jobs[index::len(devices)], results))
        while True:
            try:
                yield results.get(timeout=0.5)
            except Empty:
                if not all(future.done() for future in futures):
                    continue
                for future in futures:
                    future.result()
                # A worker may have published its final row just after the timed get.
                while True:
                    try:
                        yield results.get_nowait()
                    except Empty:
                        return


def _parallel_results(jobs, devices, args, inputs):
    # CUDA and EGL must be initialized in clean processes after binding each worker's device.
    ctx = mp.get_context("spawn")
    slots = ctx.Queue()
    try:
        for device in devices:
            slots.put(device)
        with ProcessPoolExecutor(
            max_workers=len(devices), mp_context=ctx,
            initializer=_init_worker, initargs=(slots, args, inputs),
        ) as pool:
            futures = [pool.submit(_worker_run, job) for job in jobs]
            try:
                for future in as_completed(futures):
                    yield future.result()
            finally:
                for future in futures:
                    future.cancel()
    finally:
        slots.close()
        slots.join_thread()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("retarget", "segment", "inpaint", "depth", "base_ik", "approach", "render", "composite", "curate"))
    parser.add_argument("--config", type=Path, help="batch or node benchmark YAML")
    parser.add_argument("--episodes", nargs="+", help="HDF5 files, directories, or text lists")
    parser.add_argument("--out", type=Path, help="fresh output root; episodes stored under task/name")
    parser.add_argument("--inputs-root", type=Path, nargs="+", help="upstream roots with identical task/name and piece layout")
    parser.add_argument("--robot", choices=("arx", "piper", "dual_piper"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup", type=int, help="successful episodes per worker excluded from after_warmup metrics")
    parser.add_argument("--workers", type=int, help="persistent processes; default one per visible GPU, or one for composite")
    parser.add_argument("--piece-frames", type=int)
    parser.add_argument("--inpaint-model", choices=("sttn", "propainter"))
    parser.add_argument("--window", type=int, help="depth views per window")
    args = parser.parse_args()
    try:
        cfg = read_config(args.config) if args.config else DEFAULTS
        node = cfg["node_bench"]
        roots = node["inputs_root"]
        defaults = dict(episodes=cfg["episodes"], robot=cfg["robot"], limit=cfg["limit"],
                        piece_frames=cfg["piece_frames"], workers=node["workers"], warmup=node["warmup"],
                        inputs_root=[roots] if isinstance(roots, str) else roots,
                        inpaint_model=cfg["nodes"]["inpaint"]["model"], window=cfg["nodes"]["depth"]["window"])
        if args.config:
            defaults["out"] = Path(cfg["out"]).expanduser() / cfg["name"]
            if not defaults["out"].is_absolute():
                from egowhale.paths import ROOT
                defaults["out"] = ROOT / defaults["out"]
        for key, value in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, value)
        args.ik_method, args.batch_size, args.episode_capacity = (
            ik_options(cfg["nodes"]["action"], args.robot) if args.stage == "base_ik" else ("base_ik", 0, 0)
        )
        if args.robot not in ("arx", "piper", "dual_piper"):
            raise ConfigError("robot must be arx, piper or dual_piper")
        if args.inpaint_model not in ("sttn", "propainter"):
            raise ConfigError("inpaint model must be sttn or propainter")
        if any(type(value) is not int for value in (args.limit, args.warmup, args.piece_frames, args.window)):
            raise ConfigError("limit, warmup, piece_frames and window must be integers")
    except (ConfigError, OSError, TypeError) as exc:
        parser.error(str(exc))
    if not args.episodes or args.out is None:
        parser.error("provide --episodes and --out or their equivalents in --config")
    if args.limit < 0 or args.warmup < 0 or args.piece_frames < 1 or args.window < 1:
        parser.error("invalid negative count or zero window/piece size")
    if args.workers is not None and (type(args.workers) is not int or args.workers < 1):
        parser.error("--workers must be positive")
    sources = episodes(dict(episodes=args.episodes, order=cfg["order"], seed=cfg["seed"], limit=args.limit))
    if not sources:
        parser.error("no episodes")
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error("--out must be new so prior outputs cannot be skipped or overwritten")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.stage == "composite":
        cards = [None]
    else:
        cards = [part.strip() for part in visible.split(",")]
        if any(not card.isascii() or not card.isdecimal() for card in cards):
            parser.error("set CUDA_VISIBLE_DEVICES to physical GPU indices (for example 6 or 2,6)")
        cards = [str(int(card)) for card in cards]
        if len(set(cards)) != len(cards):
            parser.error("CUDA_VISIBLE_DEVICES must list distinct GPUs")
    count = min(args.workers or len(cards), len(sources))
    devices = [cards[index % len(cards)] for index in range(count)]
    inputs = [Path(p).expanduser().resolve() for p in args.inputs_root] if args.inputs_root else None

    if count == 1:
        _bind_device(args.stage, devices[0])
        step = _make_step(args)
        if step.needs and inputs is None:
            parser.error(f"{args.stage} requires --inputs-root; upstream nodes are not run automatically")
        summary = benchmark(step, sources, out, inputs, args.warmup, args.piece_frames)
    else:
        # Check requirements before creating the pool. The parent never constructs or loads a model.
        if inputs is None:
            from egowhale.__main__ import STEPS

            cls = next(s for s in STEPS if s.name == args.stage)
            if cls.needs:
                parser.error(f"{args.stage} requires --inputs-root; upstream nodes are not run automatically")
        jobs = _jobs(sources, out)
        started = time.perf_counter()
        parallel = _parallel_stream if args.ik_method == "base_ik_seq" else _parallel_results
        summary = _record(parallel(jobs, devices, args, inputs), out, started)
    config = {
        **vars(args), "out": str(out), "episodes_resolved": [str(p) for p in sources],
        "mode": "serial_persistent_node" if count == 1 else "parallel_persistent_nodes",
        "cuda_visible_devices": visible, "workers": count, "worker_devices": devices,
    }
    (out / "run_config.json").write_text(json.dumps(config, default=str, indent=2))
    raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
