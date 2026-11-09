"""Serial batch benchmark of one node, with persistent models and explicit input roots."""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path

from egowhale.config import episodes
from egowhale.paths import IK, PREFIX
from egowhale.pieces import located, missing, pieces, plan


def prepare(src, dst, inputs, step, piece_frames):
    """Link only prerequisites; writes always belong to the fresh benchmark directory."""
    from egowhale.__main__ import STEPS

    dst.mkdir(parents=True, exist_ok=True)
    if inputs is not None and (inputs / "pieces.json").is_file():
        shutil.copyfile(inputs / "pieces.json", dst / "pieces.json")
    else:
        plan(src, dst, piece_frames)
    if step.needs:
        if inputs is None:
            raise FileNotFoundError(f"{step.name} needs {step.needs}; provide --inputs-root")
        # Carry forward known upstream artifacts, excluding this node's old outputs.
        # This lets successive isolated benchmarks be chained without copying videos.
        names = set(step.needs)
        for cls in STEPS:
            names.update(cls.makes)
        for name in sorted(names - set(step.makes)):
            for target in located(dst, name):
                source = inputs / target.relative_to(dst)
                if source.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.symlink_to(source.resolve())
        absent = missing(dst, step.needs)
        if absent:
            raise FileNotFoundError(f"Missing upstream files under {inputs}: {absent}")


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


def benchmark(step, sources, out, inputs_root=None, warmup=1, piece_frames=600):
    keys = [Path(src.parent.name) / src.stem for src in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("Episode task/name collisions or duplicates in input list")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    load_s = 0.
    loaded = False
    successes = 0
    started = time.perf_counter()
    try:
        for src, key in zip(sources, keys):
            row = dict(episode=str(src), output=str(out/key), stage=step.name, status="failed")
            episode_start = time.perf_counter()
            try:
                prepare(src, out/key, inputs_root/key if inputs_root else None, step, piece_frames)
                frames = pieces(out/key)[-1][1]
                if not loaded:
                    before = time.perf_counter()
                    step.load()
                    load_s += time.perf_counter()-before
                    loaded = True
                row.update(measure(step, src, out/key))
                row["frames"], row["frame_basis"] = processed_frames(step, out/key, frames)
                row.update(status="ok", warmup=successes<warmup)
                successes += 1
                row["fps"] = row["frames"] / max(row["node_s"], 1e-12)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            if callable(getattr(step, "compute", None)):
                # cuRobo solver reference cycles must not accumulate between episodes.
                gc.collect()
            row["episode_wall_s"] = time.perf_counter()-episode_start
            rows.append(row)
            with (out/"results.jsonl").open("a") as handle:
                handle.write(json.dumps(row)+"\n")
            summary = summarize(rows, load_s, time.perf_counter()-started)
            (out/"summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(row), flush=True)
    finally:
        step.release()
    summary = summarize(rows, load_s, time.perf_counter()-started)
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("retarget", "segment", "inpaint", "depth", "base_ik", "approach", "render", "composite", "curate"))
    parser.add_argument("--episodes", nargs="+", required=True, help="HDF5 files, directories, or text lists")
    parser.add_argument("--out", type=Path, required=True, help="fresh output root; episodes stored under task/name")
    parser.add_argument("--inputs-root", type=Path, help="upstream results root with task/name layout")
    parser.add_argument("--robot", choices=("arx", "piper", "dual_piper"), default="arx")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1, help="successful episodes excluded from after_warmup metrics")
    parser.add_argument("--piece-frames", type=int, default=600)
    parser.add_argument("--inpaint-model", choices=("sttn", "propainter"), default="sttn")
    parser.add_argument("--window", type=int, default=1, help="depth views per window")
    args = parser.parse_args()
    if args.limit < 0 or args.warmup < 0 or args.piece_frames < 1 or args.window < 1:
        parser.error("invalid negative count or zero window/piece size")
    sources = episodes(dict(episodes=args.episodes, order="list", limit=args.limit))
    if not sources:
        parser.error("no episodes")
    out = args.out.expanduser().resolve()
    if out.exists():
        parser.error("--out must be new so prior outputs cannot be skipped or overwritten")
    if args.stage != "composite":
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not device.isdigit():
            parser.error("set CUDA_VISIBLE_DEVICES to one physical GPU index (for example 9)")
        if args.stage in ("render", "curate"):
            os.environ["MUJOCO_GL"] = "egl"
            os.environ["PYOPENGL_PLATFORM"] = "egl"
            os.environ["MUJOCO_EGL_DEVICE_ID"] = device
    from egowhale.__main__ import STEPS
    from egowhale.robots.factory import make_step
    from egowhale.visual.depth import Depth
    from egowhale.visual.inpaint import Inpaint

    cls = next(s for s in STEPS if s.name == args.stage)
    step = Inpaint(args.inpaint_model) if cls is Inpaint else Depth(args.window) if cls is Depth else make_step(cls, args.robot)
    inputs = args.inputs_root.expanduser().resolve() if args.inputs_root else None
    if step.needs and inputs is None:
        parser.error(f"{args.stage} requires --inputs-root; upstream nodes are not run automatically")
    summary = benchmark(step, sources, out, inputs, args.warmup, args.piece_frames)
    (out/"run_config.json").write_text(json.dumps({**vars(args), "out": str(out),
        "episodes_resolved": [str(p) for p in sources], "mode": "serial_persistent_node", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}, default=str, indent=2))
    raise SystemExit(1 if summary["failed"] else 0)


if __name__ == "__main__":
    main()
