"""Tasks executed by the standalone WiLoR worker pools."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from egowhale.wilor.pipeline.hmp import merge_windows
from egowhale.wilor.synced import SYNC_ROOT, identity, publish

CANDIDATE_FIELDS = {
    "bbox": ((4,), np.float32),
    "is_right": ((), np.bool_),
    "score": ((), np.float64),
    "keypoints_3d": ((21, 3), np.float32),
    "cam_t": ((3,), np.float32),
    "focal_length": ((), np.float64),
    "mano_pose": ((48,), np.float32),
    "mano_shape": ((10,), np.float32),
}


def rgb_frames(path):
    import av

    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def save_candidates(frames, directory, count, keep_all, block_size=128):
    """Store numeric chunk files and per-frame offsets, including empty frames."""
    directory.mkdir()
    rows, offsets, written = [], [0], 0
    for hands in frames:
        if not keep_all:
            best = {}
            for hand in hands:
                if hand.is_right not in best or hand.score > best[hand.is_right].score:
                    best[hand.is_right] = hand
            hands = best.values()
        rows.extend(hands)
        offsets.append(len(rows))
        written += 1
        if len(offsets) == block_size + 1 or written == count:
            arrays = {name: np.asarray([getattr(row, name) for row in rows], dtype=dtype)
                      .reshape((len(rows), *shape)) for name, (shape, dtype) in CANDIDATE_FIELDS.items()}
            np.savez(directory / f"{written:08d}.npz", offsets=offsets, **arrays)
            rows, offsets = [], [0]
    if written != count:
        raise ValueError(f"WiLoR decoded {written} frames, expected {count}")


def read_candidates(directory, count):
    read = 0
    for block in sorted(directory.glob("*.npz")):
        with np.load(block, allow_pickle=False) as data:
            arrays = dict(data)
        offsets = arrays.pop("offsets")
        for a, b in zip(offsets[:-1], offsets[1:]):
            read += 1
            yield [SimpleNamespace(**{key: value[i] for key, value in arrays.items()}) for i in range(a, b)]
    if read != count:
        raise ValueError(f"Candidate timeline contains {read} frames, expected {count}")


def optimize_command(cfg, work, args, timeout=None):
    from ego2robot.utils.progress import run_streamed_command

    repo = Path(cfg.dynhamr_root)
    launcher = SYNC_ROOT / "scripts/run_dynhamr.py"
    proc = run_streamed_command(
        [cfg.dynhamr_python, str(launcher), "--dynhamr-root", str(repo), *args],
        cwd=repo / "dyn-hamr", stdout_path=work / "stdout.log", stderr_path=work / "stderr.log",
        timeout=timeout or cfg.dynhamr_timeout, progress_only=True,
    )
    if proc.returncode:
        raise RuntimeError(f"DynHaMR exited {proc.returncode}: {proc.stderr[-1000:]}")


def prepare_optimizer(processor, work, attempt, video, camera, count, height, width):
    # Mirror the official automatic track selection. Read the constant without importing CUDA code.
    import ast
    tree = ast.parse((Path(processor.cfg.dynhamr_root) / "dyn-hamr/data/dataset.py").read_text())
    min_track_len = next(ast.literal_eval(node.value) for node in tree.body
                         if isinstance(node, ast.Assign) and
                         any(isinstance(t, ast.Name) and t.id == "MIN_TRACK_LEN" for t in node.targets))
    heads = work / "hand_processor"
    counts = []
    for side in ("left", "right"):
        with np.load(heads / f"hand_data_{side}.npz") as data:
            counts.append(len(data["detected_frame_ids"]))
    if max(counts) <= min_track_len:
        (attempt / "skipped.json").write_text(json.dumps({"reason": "no_eligible_tracks", "counts": counts}))
        return

    cfg = processor.cfg
    inputs = attempt / "inputs"
    with closing(rgb_frames(video)) as frames:
        processor._prepare_dynhamr_images(frames, str(inputs), "wilor")
    processor._convert_to_dynhamr(heads, str(inputs), "wilor", *camera,
                                  cfg.get("anyhand_focal", 2500), count, height, width)
    # Upstream HMP uses the video path for metadata and its filename.
    (inputs / "video").mkdir()
    (inputs / "video" / "wilor.mp4").symlink_to(video)


def root_smooth(cfg, inputs, attempt, fps):
    from ego2robot.utils.progress import report_progress

    output = attempt / "smooth"
    report_progress("DynHaMR root / smooth")
    optimize_command(cfg, attempt, [
        "--prepare-hmp", str(output), "data=video_vipe", "run_opt=True",
        "run_prior=True", "run_vis=False", "is_static=True", "overwrite=True", "gpu=0",
        "data.seq=wilor", f"data.root={json.dumps(str(inputs), ensure_ascii=False)}",
        f"paths.DATA_DIR={json.dumps(str(Path(cfg.dynhamr_root) / '_DATA/data'), ensure_ascii=False)}",
        f"hydra.run.dir={json.dumps(str(output), ensure_ascii=False)}",
    ])
    status_path = output / "smooth_completed.json"
    status = json.loads(status_path.read_text())
    status["fps"] = fps
    status["inputs"] = str(inputs)
    status_path.write_text(json.dumps(status))


def finalize_optimization(processor, request, work, camera, count, height, width):
    cfg = processor.cfg
    save_dir = work / "hand_processor"
    smooth = Path(request["smooth"]) if request.get("smooth") else None
    gap = int(cfg.get("dynhamr_gap_fill", 3))
    z_min, z_max = float(cfg.get("dynhamr_z_min", .15)), float(cfg.get("dynhamr_z_max", .4))
    reason = request.get("fallback_reason")
    if smooth:
        checkpoint = json.loads((smooth / "smooth_completed.json").read_text())
        kwargs = (str(cfg.dynhamr_root), checkpoint["inputs"], "wilor", save_dir, *camera, count)
        processor._smooth_result_path = checkpoint["smooth_result"]
        start = checkpoint["frame_start"]
        if not reason:
            try:
                windows = request["windows"]
                if windows[0]["hand"] == -1:
                    result_file = windows[0]["result"]
                else:
                    chunks = []
                    for w in sorted(windows, key=lambda item: (item["hand"], item["start"])):
                        with np.load(w["result"]) as data:
                            chunks.append((w["hand"], w["start"], w["stop"], dict(data)))
                    with np.load(checkpoint["smooth_result"]) as data:
                        merged = merge_windows(dict(data), chunks)
                    result_file = work / "hmp_merged.npz"
                    np.savez_compressed(result_file, **merged)
                hands = processor._postprocess_dynhamr(*kwargs, gap, z_min, z_max, result_path=result_file,
                                                       frame_start=start, write_output=False)
                if not hands:
                    raise ValueError("HMP postprocess produced no hand arrays")
            except Exception as exc:
                reason = f"Invalid HMP result: {exc}"
            else:
                processor._write_dynhamr_hands(save_dir, hands)
                return {"result_source": "hmp", "succeeded": True, "fallback": False,
                        "hmp_succeeded": True, "root_smooth_completed": True,
                        "masks_available": request.get("masks_available", True)}
        quality = processor._try_smooth_fallback(*kwargs, height, width, gap, z_min, z_max, frame_start=start)
    else:
        quality = None
    accepted = bool(quality and quality["accepted"])
    return {"result_source": "smooth_fit" if accepted else "wilor_filtered",
            "succeeded": accepted, "fallback": True, "reason": reason, "quality": quality,
            "root_smooth_completed": bool(smooth), "hmp_succeeded": False,
            "masks_available": request.get("masks_available", True)}


def execute(request, experiment, models):
    from ego2robot.processors.hand_processor import HandProcessor
    from ego2robot.utils.progress import report_progress

    stage = request["stage"]
    source = Path(request["video"])
    dst = Path(request["output"])
    work = dst / "action/wilor_ego2robot"
    attempt = Path(request["attempt_dir"])
    attempt.mkdir(parents=True, exist_ok=True)
    cfg = experiment.load_config(Path(request["config"]), work)
    cfg.dynhamr_gpu = 0
    if stage == "check":
        errors = experiment.check_requirements(cfg, "da3", "full")
        if errors:
            raise RuntimeError("\n".join(errors))
        return
    video = work / "video_L.mp4"
    if stage == "prepare":
        if identity(source) != request["identity"]:
            raise ValueError("Source changed since job admission")
        experiment.prepare_video(source, attempt, 0, int(cfg.get("input_resolution", 480)))
        if identity(source) != request["identity"]:
            raise ValueError("Source changed while preparing video")
        work.mkdir(parents=True, exist_ok=True)
        for name in ("video_L.mp4", "input.json"):
            (attempt / name).replace(work / name)
        return

    info = json.loads((work / "input.json").read_text())
    width, height = info["size"]
    count, fps = info["frames"], info["fps"]
    if stage == "camera":
        (attempt / "video_L.mp4").symlink_to(video)
        shutil.copy2(work / "input.json", attempt)
        cfg.da3_max_views = request["max_views"]
        cfg.save_depth_maps = cfg.get("save_depth_maps", False)
        cfg.known_intrinsics = False
        cfg.camera_intrinsics = cfg.camera_extrinsics = ""
        models.depth(cfg, attempt)
        for name in ("cam_intrinsics.json", "cam_extrinsics.npz", "da3_samples.json"):
            (attempt / name).replace(work / name)
        if cfg.save_depth_maps:
            if (work / "depth").exists():
                shutil.rmtree(work / "depth")
            (attempt / "depth").replace(work / "depth")
        return

    processor = HandProcessor(cfg)
    hands = work / "hand_processor"
    if stage == "candidates":
        processor._wrapper = models.hand_wrapper
        processor._ensure_model()
        models.hand_wrapper = processor._wrapper
        with closing(rgb_frames(video)) as frames:
            save_candidates(processor._wrapper.candidates(frames, count), attempt / "candidates", count,
                            keep_all=cfg.dynhamr_enabled)
        return
    if stage == "masks":
        from ego2robot.processors.wilor_masks import build_predictor, propagate_masks
        if getattr(models, "mask_predictor", None) is None:
            report_progress("加载 SAM3")
            models.mask_predictor = build_predictor(cfg.sam3_checkpoint)
        propagate_masks(models.mask_predictor, video, attempt / "masks.npy", (count, height, width),
                        request["max_mask_frames"])
        return
    if stage == "tracks":
        hands.mkdir(exist_ok=True)
        masks = np.load(request["masks"], mmap_mode="r") if request.get("masks") else None
        cfg.hand_mode = "full"
        processor.save_tracks(read_candidates(Path(request["candidates"]), count), masks, hands, count, height, width)
        return
    if stage == "hmp":
        report_progress("HMP 运动先验", message=f"窗口 [{request['start']}, {request['stop']})，手 {request['hand']}")
        optimize_command(cfg, attempt, [
            "--hmp-checkpoint", request["smooth"], "--hmp-output", str(attempt / "window"),
            "--hmp-window", str(request["hand"]), str(request["start"]), str(request["stop"]),
        ])
        return
    if stage == "raw_video":
        with closing(rgb_frames(video)) as frames:
            processor._save_hand_overlay_video(frames, hands, None, height, width,
                                               suffix="_anyhand", data_prefix="hand_data_raw",
                                               fps=fps, frame_count=count)
        return

    camera = processor._load_intrinsics(str(work))
    if stage == "basic":
        from ego2robot.models.anyhand_wrapper import AnyHandWrapper

        fx, fy, cx, cy = camera
        focal = (fx + fy) / 2
        hands.mkdir(exist_ok=True)
        result = AnyHandWrapper.calibrate(read_candidates(Path(request["candidates"]), count), focal)
        processor.save_basic(result, hands, fx, fy, cx, cy, count, height, width, fps)
    elif stage == "optimizer_input":
        prepare_optimizer(processor, work, attempt, video, camera, count, height, width)
    elif stage == "smooth":
        root_smooth(cfg, Path(request["inputs"]), attempt, fps)
    elif stage == "merge":
        # Restore the raw baseline before (re)applying optimization results.
        for side in ("left", "right"):
            shutil.copy2(hands / f"hand_data_raw_{side}.npz", hands / f"hand_data_{side}.npz")
        status = finalize_optimization(processor, request, work, camera, count, height, width)
        (work / "optimization_status.json").write_text(json.dumps(status, ensure_ascii=False))
    elif stage == "publish":
        if identity(source) != request["identity"]:
            raise ValueError("Source changed before publishing")
        status_path = work / "optimization_status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"result_source": "wilor", "fallback": False}
        processor.optimization_status = status
        with closing(rgb_frames(video)) as frames:
            processor._save_hand_overlay_video(frames, hands, None, height, width, fps=fps, frame_count=count)
        experiment.finish(work, "full", "b", "da3", status, expected_frames=count)
        publish(source, dst, work, request["signature"], retarget_hdf5=request["retarget_hdf5"])
        from egowhale.wilor.step import validate_prediction_input
        validate_prediction_input(source, dst)
    else:
        raise ValueError(f"Unknown parallel stage: {stage}")
