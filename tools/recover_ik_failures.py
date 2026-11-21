"""Retry rejected sequential IK episodes and export their TensorBoard comparisons."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def recover_shard(device, run, prepared, output, cfg, rows):
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    os.environ["MUJOCO_GL"] = os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ["MUJOCO_EGL_DEVICE_ID"] = device
    os.environ.pop("EGOWHALE_VLM_API_KEY", None)
    egl_vendor = run / "egl_nvidia.json"
    if egl_vendor.is_file():
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl_vendor)

    import numpy as np

    from egowhale.action import base_ik as core
    from egowhale.action.approach import Approach
    from egowhale.action.base_ik_seq import EpisodeStream
    from egowhale.action.sequential_ik import CuroboBackend
    from egowhale.media import video_shape
    from egowhale.paths import COMPOSITE, DEPTH, GRIPPER, INPAINT, MASKS, QUALITY
    from egowhale.pieces import located, pieces
    from egowhale.quality.curate import Curate
    from egowhale.step import run as run_steps
    from egowhale.visual.composite import Composite
    from egowhale.visual.render import Render

    class RecoveryBackend(CuroboBackend):
        def base_candidates(self, offsets, position, valid, group, seed_r, seed_t, geometry):
            try:
                return super().base_candidates(offsets, position, valid, group, seed_r, seed_t, geometry)
            except core.NoBaseCandidate:
                box, intrinsic, size = geometry
                bases = np.stack([core._camera_matrix(seed_r, seed_t, offset) for offset in offsets])
                # Keep the original distance margin if a geometrically reachable base only failed visibility.
                reachable = core.candidate_mask(bases, position, valid, group.reach, None, intrinsic, size)
                if reachable.any():
                    policies[group.name] = "allow_visible_base"
                    return reachable, (None, intrinsic, size)
                # Keep visibility on the remaining cases while using the model's nominal reach.
                reachable = core.candidate_mask(bases, position, valid, group.reach / 0.9, box, intrinsic, size)
                if reachable.any():
                    policies[group.name] = "nominal_reach"
                    return reachable, geometry
                raise

    solver = core.BaseIK(cfg["robot"])
    batch_size = sum(len(core._offset_grid(group.bounds)) for group in solver.robot.groups)
    stream = EpisodeStream(solver.robot, batch_size, backend=RecoveryBackend(solver.robot, 1))
    steps = (Render(cfg["robot"]), Composite(), Curate(cfg["robot"]))
    if cfg["approach"]:
        steps = (Approach(cfg["robot"]), *steps)
    completed = []
    try:
        for original, src in rows:
            name = original["episode"]
            dst = output / name
            policies = {}
            result = {"episode": name, "original_error": original["error"], "policy": policies}
            try:
                dst.mkdir(parents=True, exist_ok=False)
                for root, filename in (
                    (run, "pieces.json"), (run, GRIPPER),
                    (prepared / "segment", MASKS), (prepared / "inpaint", INPAINT),
                    (prepared / "depth", DEPTH),
                ):
                    paths = located(root / name, filename)
                    outputs = located(dst, filename)
                    if filename in (MASKS, INPAINT, DEPTH) and pieces(root / name) != pieces(dst):
                        raise ValueError(f"incompatible piece layout for {filename}")
                    for src_file, dst_file in zip(paths, outputs, strict=True):
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        dst_file.symlink_to(src_file.resolve(strict=True))
                outcomes = stream.advance([(name, solver.read(src, dst))])
                while not outcomes:
                    outcomes = stream.advance()
                outcome = outcomes[0]
                if not outcome["ok"]:
                    raise RuntimeError(outcome["error"])
                solver.save(dst, outcome["out"])
                run_steps(src, dst, steps)
                quality = json.loads((dst / QUALITY).read_text())
                frames, _, _, fps = video_shape(dst / COMPOSITE)
                source_frames, _, _, source_fps = video_shape(src.with_suffix(".mp4"))
                if (not cfg["approach"] and frames != source_frames) or not np.isclose(fps, source_fps):
                    raise ValueError(f"unexpected composite format: {frames} frames, {fps} FPS")
                result.update(ok=True, frames=frames, fps=fps, curate_keep=quality["keep"],
                              invalid_ratio=quality["invalid_ratio"], mean_ik_error_m=quality["mean_ik_error_m"])
            except Exception as exc:
                result.update(ok=False, error=f"{type(exc).__name__}: {exc}")
            (dst / "recovery.json").write_text(json.dumps(result, indent=2) + "\n")
            completed.append(result)
            print(json.dumps(result), flush=True)
    finally:
        stream.close()
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Batch output directory with config.yaml and batch.jsonl")
    parser.add_argument("--prepared", type=Path, required=True, help="Prepared segment, inpaint, and depth roots")
    parser.add_argument("--gpus", required=True, help="Comma-separated physical GPU indices")
    args = parser.parse_args()

    import yaml

    from egowhale.config import episodes

    run = args.run.resolve(strict=True)
    prepared = args.prepared.resolve(strict=True)
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    if cfg["nodes"]["action"]["ik_method"] != "base_ik_seq":
        parser.error("the rejected episodes must come from a sequential Base+IK run")
    devices = args.gpus.split(",")
    if any(not card.isdecimal() for card in devices) or len(set(devices)) != len(devices):
        parser.error("provide distinct numeric GPU indices")
    sources = {f"{p.parent.name}/{p.stem}": p for p in episodes(cfg)}
    rows = []
    for line in (run / "batch.jsonl").read_text().splitlines():
        row = json.loads(line)
        if not row["ok"] and row["error"].startswith("base_ik:") and "no offscreen, reachable base" in row["error"]:
            rows.append((row, sources[row["episode"]]))
    if not rows:
        parser.error("no Base+IK geometry rejections found")
    output = run / "ik_recovery"
    board = run / "tb_ik_failed"
    if output.exists() or board.exists():
        parser.error("recovery outputs already exist")
    output.mkdir()
    with ProcessPoolExecutor(max_workers=len(devices), mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(recover_shard, device, run, prepared, output, cfg, rows[i::len(devices)])
                   for i, device in enumerate(devices) if rows[i::len(devices)]]
        recovered = {result["episode"]: result for future in futures for result in future.result()}

    from torch.utils.tensorboard import SummaryWriter

    from egowhale.batch.dag import Job
    from egowhale.batch.report import write_comparison

    with SummaryWriter(log_dir=str(board)) as writer:
        for index, (row, src) in enumerate(rows):
            result = recovered[row["episode"]]
            if result["ok"]:
                job = Job(src, output / row["episode"], cfg["piece_frames"], approach=cfg["approach"])
                write_comparison(job, writer, index)
    (output / "results.jsonl").write_text("".join(json.dumps(recovered[row["episode"]]) + "\n" for row, _ in rows))
    return int(not all(result["ok"] for result in recovered.values()))


if __name__ == "__main__":
    raise SystemExit(main())
