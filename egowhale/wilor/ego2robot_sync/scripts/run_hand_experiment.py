#!/usr/bin/env python3
"""Prepare one video and run SAM3 + AnyHand/WiLoR + DynHaMR/HMP."""
import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
THIRDPARTY = Path(os.environ.get('EGOWHALE_WILOR_THIRDPARTY', ROOT.parents[2] / 'thirdparty')).resolve()
sys.path.insert(0, str(ROOT))
from ego2robot.utils.progress import report_progress


def load_config(path, output):
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    for key in ("anyhand_root", "dynhamr_root", "sam3_checkpoint", "da3_model_root"):
        configured = Path(cfg[key])
        cfg[key] = str((THIRDPARTY / configured.relative_to('thirdparty')).resolve() if not configured.is_absolute() and configured.parts[0] == 'thirdparty' else (ROOT / configured).resolve())
    configured_python = cfg.get("dynhamr_python")
    cfg.dynhamr_python = str((ROOT / configured_python).resolve()) if configured_python else sys.executable
    cfg.data_root_dir = cfg.processed_data_root_dir = str(output.parent)
    cfg.demo_name = output.name
    cfg.dynhamr_log_dir = str(output)
    cfg.sam3_hand_masks = str(output / "hand_masks" / "hand_masks.npy")
    return cfg


def check_da3_import():
    """Import the inference API without loading weights or allocating GPU memory."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "from depth_anything_3.api import DepthAnything3; print('DA3 API import OK')"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "DA3 推理接口导入超时（120 秒）"
    if result.returncode:
        detail = (result.stderr or result.stdout)[-2000:].strip()
        return f"DA3 推理接口导入失败：{detail}"
    print("DA3 推理接口导入通过", flush=True)
    return None


def check_requirements(cfg, camera_source, stage, needs_hand=True):
    errors = []
    use_optimizer = needs_hand and bool(cfg.get("dynhamr_enabled", False)) and stage != "detect"
    main_modules = ["torch", "cv2", "av", "omegaconf", "natsort", "mediapy", "IPython", "scipy", "smplx", "chumpy"]
    if use_optimizer:
        main_modules += ["tensorboard", "human_body_prior"]
    if camera_source == "egodex":
        main_modules.append("h5py")
    if needs_hand and stage != "refine":
        main_modules += ["pytorch_lightning", "ultralytics", "yacs", "pyrender", "timm"]
        if cfg.get("dynhamr_enabled", False):
            main_modules.append("sam3")
    if camera_source == "da3":
        main_modules.append("depth_anything_3")
    for name in main_modules:
        if importlib.util.find_spec(name) is None:
            errors.append(f"主环境缺少 Python 模块：{name}")
    if camera_source == "da3":
        import_error = check_da3_import()
        if import_error:
            errors.append(import_error)
    anyhand = Path(cfg.anyhand_root)
    dyn = Path(cfg.dynhamr_root)
    files = []
    if use_optimizer:
        files += [dyn / "dyn-hamr/run_opt.py", dyn / "_DATA/data/mano/MANO_RIGHT.pkl",
                  dyn / "_DATA/data/mano_mean_params.npz"]
    if needs_hand and stage != "refine":
        files += [anyhand / "scripts/rgb_predictor.py",
                  anyhand / "WiLoR/wilor/models/__init__.py",
                  anyhand / "pretrained_models/anyhand_wilor.ckpt",
                  anyhand / "pretrained_models/model_config_wilor.yaml",
                  anyhand / "pretrained_models/detector.pt",
                  anyhand / "mano_data/MANO_RIGHT.pkl"]
        if cfg.get("dynhamr_enabled", False):
            files.append(Path(cfg.sam3_checkpoint))
    prior_files = [dyn / "_DATA/hmp_model/results/model" / name
                   for name in ("local_encoder.pth", "global_encoder.pth", "nemf.pth")]
    prior_files += [dyn / "_DATA/hmp_model" / name
                    for name in ("mean-neutral-128-30fps.pt", "std-neutral-128-30fps.pt")]
    missing_prior = [str(path) for path in prior_files if not path.is_file() or path.stat().st_size == 0]
    if use_optimizer and missing_prior:
        print("HMP 资源不齐，先验失败时将回退 WiLoR 结果：" + ", ".join(missing_prior), flush=True)
    if use_optimizer:
        for name in ("bone_len_max.npy", "bone_len_min.npy", "CONVEX_HULLS.npy", "curvatures_max.npy",
                     "curvatures_min.npy", "PHI_max.npy", "PHI_min.npy"):
            files.append(dyn / "_DATA/BMC" / name)
    if camera_source == "da3":
        from ego2robot.models.da3_wrapper import _MODEL_DIRS
        model_dir = _MODEL_DIRS.get(cfg.depth_model)
        if model_dir is None:
            errors.append(f"未知 DA3 模型：{cfg.depth_model}")
            model_dir = str(cfg.depth_model)
        files += [Path(cfg.da3_model_root) / model_dir / name
                  for name in ("config.json", "model.safetensors")]
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"缺少资源：{path}")
    import shutil
    if not shutil.which("ffmpeg"):
        errors.append("PATH 中没有 ffmpeg，请激活实验环境")
    # The optimizer defaults to this same environment; avoid duplicate CUDA probes.
    cuda_code = "import torch; print(torch.__version__); assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.cuda.get_device_name(0))"
    interpreters = [("实验环境", sys.executable)]
    if use_optimizer and Path(cfg.dynhamr_python).resolve() != Path(sys.executable).resolve():
        interpreters.append(("自定义优化器解释器", cfg.dynhamr_python))
    for label, executable in interpreters:
        if not Path(executable).is_file():
            errors.append(f"{label}解释器不存在：{executable}")
            continue
        result = subprocess.run([str(executable), "-c", cuda_code], capture_output=True, text=True)
        if result.returncode:
            errors.append(f"{label} GPU 检查失败：{(result.stderr or result.stdout)[-400:].strip()}")
        else:
            print(label, result.stdout.strip())
    if use_optimizer and Path(cfg.dynhamr_python).is_file():
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(dyn / "dyn-hamr"), str(dyn / "dyn-hamr/HMP"),
                                             str(THIRDPARTY / "human_body_prior/src")])
        # The compatibility entry prepares legacy MANO imports before loading DynHaMR.
        result = subprocess.run([cfg.dynhamr_python, str(ROOT / "scripts/run_dynhamr.py"),
                                 "--dynhamr-root", str(dyn), "--check-import"], cwd=dyn / "dyn-hamr",
                                env=env, capture_output=True, text=True, timeout=120)
        if result.returncode:
            errors.append(f"DynHaMR 优化器导入失败：{result.stderr[-700:].strip()}")
    return errors


def resolve_inputs(video, path_mode, camera_source, intrinsics_h5=None, intrinsics_json=None, video_only=False):
    from ego2robot.utils.egodex_path_a import inspect_episode
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(video)
    h5 = Path(intrinsics_h5) if intrinsics_h5 else video.with_suffix('.hdf5')
    if intrinsics_h5 and not h5.is_file():
        raise FileNotFoundError(h5)
    if intrinsics_json and not Path(intrinsics_json).is_file():
        raise FileNotFoundError(intrinsics_json)
    if video_only and (intrinsics_h5 or intrinsics_json or path_mode == 'a' or camera_source in ('egodex', 'provided')):
        raise ValueError('video-only 只使用视频，不能同时指定标注或相机内参')
    has_k, has_labels = (False, False) if video_only else inspect_episode(h5)
    if path_mode == 'a' and not has_labels:
        raise ValueError(f'Path A 需要 EgoDex 手部标注：{h5}')
    selected_path = ('a' if has_labels else 'b') if path_mode == 'auto' else path_mode
    selected_camera = ('provided' if intrinsics_json else 'egodex' if has_k else 'da3') if camera_source == 'auto' else camera_source
    if selected_camera == 'provided' and not intrinsics_json:
        raise ValueError('provided 相机参数需要 --intrinsics-json')
    if selected_camera == 'egodex' and not has_k:
        raise ValueError(f'HDF5 缺少 camera/intrinsic：{h5}')
    return selected_path, selected_camera, h5


def prepare_video(video, output, camera_source, intrinsics_h5, max_frames, height,
                  intrinsics_json=None, processing_path=None):
    import av
    import cv2
    import numpy as np
    if not video.is_file():
        raise FileNotFoundError(video)
    if camera_source == "egodex" and not intrinsics_h5.is_file():
        raise FileNotFoundError(intrinsics_h5)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "video_L.mp4"
    manifest = output / "input.json"
    intrinsics_source = str(intrinsics_h5) if camera_source == "egodex" else None
    request = {"source": str(video), "height": height, "max_frames": max_frames,
               "camera_source": camera_source, "intrinsics_h5": intrinsics_source,
               "intrinsics_json": str(intrinsics_json) if camera_source == 'provided' else None,
               "processing_path": processing_path,
               "annotation_h5": str(intrinsics_h5) if processing_path == 'a' else None}
    if target.exists() or manifest.exists():
        if not manifest.exists() or json.loads(manifest.read_text()).get("request") != request:
            raise RuntimeError("输出目录已有其他输入或配置，请选择新的 --output 目录")
        print(f"复用已准备的输入：{target}")
        return
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 30)
        writer = None
        count = 0
        expected_frames = min(stream.frames, max_frames) if stream.frames and max_frames else (stream.frames or max_frames or None)
        report_progress("输入视频准备", 0, expected_frames)
        try:
            for frame in container.decode(stream):
                if max_frames and count >= max_frames:
                    break
                image = frame.to_ndarray(format="bgr24")
                if writer is None:
                    orig_h, orig_w = image.shape[:2]
                    new_h = min(height, orig_h) // 2 * 2
                    new_w = round(orig_w * new_h / orig_h) // 2 * 2
                    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (new_w, new_h))
                    if not writer.isOpened():
                        raise RuntimeError(f"无法写入 {target}")
                writer.write(cv2.resize(image, (new_w, new_h)))
                count += 1
                if count % 30 == 0:
                    report_progress("输入视频准备", count, expected_frames)
        finally:
            if writer is not None:
                writer.release()
    if count == 0:
        target.unlink(missing_ok=True)
        raise ValueError('输入视频没有可解码帧')
    if camera_source in ('egodex', 'provided'):
        if camera_source == 'egodex':
            import h5py
            with h5py.File(intrinsics_h5, "r") as handle:
                K = np.asarray(handle["camera/intrinsic"], dtype=float)
            ref_w, ref_h = orig_w, orig_h
        else:
            data = json.loads(Path(intrinsics_json).read_text())
            camera = data.get('left', data)
            K = np.asarray(data['K'], dtype=float) if 'K' in data else np.array(
                [[camera['fx'], 0, camera['cx']], [0, camera.get('fy', camera['fx']), camera['cy']], [0, 0, 1]], dtype=float)
            ref_w = camera.get('new_w', camera.get('width', data.get('width', orig_w)))
            ref_h = camera.get('new_h', camera.get('height', data.get('height', orig_h)))
        if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or ref_w <= 0 or ref_h <= 0:
            raise ValueError('相机内参无效')
        K[0] *= new_w / ref_w
        K[1] *= new_h / ref_h
        intrinsics = {"fps": fps, "left": {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                      "cx": float(K[0, 2]), "cy": float(K[1, 2]), "new_w": new_w,
                      "new_h": new_h, "orig_w": orig_w, "orig_h": orig_h, "fps": fps}}
        (output / "cam_intrinsics.json").write_text(json.dumps(intrinsics, indent=2))
    manifest.write_text(json.dumps({"request": request, "frames": count, "fps": fps,
                                   "size": [new_w, new_h], "hand_labels_from_hdf5": False}, indent=2))
    print(f"已准备 {count} 帧，{new_w}x{new_h}，{fps:g} FPS；" +
          ("下一阶段读取 HDF5 手部标注" if processing_path == 'a' else "手部使用 WiLoR 预测"))
    report_progress("输入视频准备", count, count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hand_experiment.yaml")
    parser.add_argument("--video", type=Path, default=ROOT / "datasets/egodex/test/add_remove_lid/1.mp4")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/hand_experiment/add_remove_lid_1")
    parser.add_argument("--path", choices=['auto', 'a', 'b'], default='auto', help='auto: 有手部标注走 A，否则走 WiLoR B')
    parser.add_argument("--camera-source", choices=["auto", "egodex", "provided", "da3"], default="auto")
    parser.add_argument("--intrinsics-h5", type=Path)
    parser.add_argument("--intrinsics-json", type=Path)
    parser.add_argument("--video-only", action="store_true", help="只读取视频，忽略同名 HDF5 并由 DA3 估计内参")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--stage", choices=["full", "detect", "refine"], default="full")
    parser.add_argument("--dynhamr-python", type=Path)
    parser.add_argument("--enable-dynhamr", action="store_true", help="Path B 启用 DynHaMR 和原始 HMP")
    parser.add_argument("--depth-stride", type=int)
    parser.add_argument("--final-only", action="store_true", help="输出优化后标注视频，省略优化前预览与深度图文件")
    parser.add_argument("--prechecked", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--check", action="store_true", help="检查统一实验环境、模型和 GPU")
    parser.add_argument("--prepare-only", action="store_true", help="只准备视频和相机内参，不运行模型")
    args = parser.parse_args()
    if args.height < 2 or args.max_frames < 0:
        parser.error("height 必须至少为 2，max-frames 必须非负（0 表示全片）")
    if args.depth_stride is not None and args.depth_stride < 1:
        parser.error("depth-stride 必须为正整数")
    if args.final_only and args.stage == "detect":
        parser.error("final-only 需要 full 或 refine 阶段")
    output, video = args.output.resolve(), args.video.resolve()
    h5_input = args.intrinsics_h5.resolve() if args.intrinsics_h5 else None
    input_is_h5 = video.suffix.lower() in ('.hdf5', '.h5')
    if video.suffix.lower() in ('.hdf5', '.h5'):
        if h5_input and h5_input != video:
            parser.error('HDF5 输入与 --intrinsics-h5 不能指向不同文件')
        h5_input = video
        video = video.with_suffix('.mp4')
    json_input = args.intrinsics_json.resolve() if args.intrinsics_json else None
    if args.path == 'auto' and not input_is_h5 and h5_input is None:
        args.path = 'b'
    path_mode, camera_source, h5 = resolve_inputs(video, args.path, args.camera_source, h5_input, json_input, args.video_only)
    h5 = h5.resolve()
    cfg = load_config(args.config, output)
    if args.enable_dynhamr:
        cfg.dynhamr_enabled = True
        if path_mode == 'a':
            print('Path A 直接使用手部标注；DynHaMR 开关仅作用于 Path B', flush=True)
    if args.stage == 'refine' and not cfg.dynhamr_enabled:
        parser.error('refine 阶段需要 --enable-dynhamr 或配置 dynhamr_enabled: true')
    if args.dynhamr_python:
        cfg.dynhamr_python = str(args.dynhamr_python.resolve())
    if args.depth_stride is not None:
        cfg.depth_stride = args.depth_stride
    cfg.dynhamr_gpu = 0  # logical GPU 0 inside CUDA_VISIBLE_DEVICES
    cfg.save_raw_overlay = not args.final_only
    cfg.save_depth_maps = not args.final_only
    if camera_source == "da3":
        cfg.known_intrinsics = False
        cfg.camera_intrinsics = cfg.camera_extrinsics = ""
    if not args.prepare_only and (not args.prechecked or args.check):
        report_progress("运行条件检查")
        errors = [] if path_mode == 'a' and camera_source != 'da3' else check_requirements(
            cfg, camera_source, args.stage, needs_hand=path_mode == 'b')
        if errors:
            print("尚不能运行完整优化：\n" + "\n".join(f"- {e}" for e in errors), file=sys.stderr)
            return 1
        if args.check:
            print("视频处理所需检查通过（HMP 失败允许回退）")
            return 0
    print(f"处理路径：Path {path_mode.upper()} | 相机参数：{camera_source}", flush=True)
    prepare_video(video, output, camera_source, h5, args.max_frames, args.height, json_input, path_mode)
    if args.prepare_only:
        return 0
    marker = output / f"{args.stage}_completed.json"
    marker.unlink(missing_ok=True)
    from omegaconf import OmegaConf
    da3_marker = output / "da3_completed.json"
    if camera_source == "da3" and not (da3_marker.exists() and (output / "cam_intrinsics.json").exists() and (output / "cam_extrinsics.npz").exists()):
        report_progress("DA3 相机估计", message="由视频估计相机参数，不读取 HDF5")
        from ego2robot.processors.depth_pose_processor import DepthPoseProcessor
        depth_processor = DepthPoseProcessor(cfg)
        try:
            depth_processor.process_one_demo(str(output))
        finally:
            if depth_processor._model is not None:
                depth_processor._model.release()
                depth_processor._model = None
        da3_marker.write_text(json.dumps({"camera_source": "da3"}))
    from ego2robot.processors.hand_processor import HandProcessor
    cfg.hand_mode = {"full": "full", "detect": "detect_only", "refine": "dynhamr_only"}[args.stage]
    processor = HandProcessor(cfg)
    (output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    if path_mode == 'a':
        import numpy as np
        from ego2robot.utils.egodex_path_a import read_hands
        frames, fps = processor._load_frames(str(output))
        camera = json.loads((output / 'cam_intrinsics.json').read_text())['left']
        K = np.array([[camera['fx'], 0, camera['cx']], [0, camera['fy'], camera['cy']], [0, 0, 1]])
        hands = read_hands(h5, len(frames), K)
        save_dir = output / 'hand_processor'
        save_dir.mkdir(exist_ok=True)
        for side, (xyz, uv, valid) in hands.items():
            np.savez_compressed(save_dir / f'hand_data_{side}.npz', kpts_3d=xyz, kpts_2d=uv,
                                hand_detected=valid, frame_indices=np.arange(len(frames)))
        processor._save_hand_overlay_video(frames, save_dir, (camera['fx'] + camera['fy']) / 2,
                                           frames[0].shape[0], frames[0].shape[1], fps=fps)
        manifest_path = output / 'input.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['hand_labels_from_hdf5'] = True
        manifest_path.write_text(json.dumps(manifest, indent=2))
        optimization = {'result_source': 'egodex_annotations', 'fallback': False}
    else:
        processor.process_one_demo(str(output))
        optimization = getattr(processor, "optimization_status", {'result_source': 'wilor', 'fallback': False})
    final_video = None
    if args.stage != "detect":
        final_video = output / "optimized_hand_points.mp4"
        source_video = output / "hand_processor/video_hand_overlay.mp4"
        if not source_video.is_file() or source_video.stat().st_size == 0:
            raise RuntimeError("未生成有效手部标注视频")
        import av
        import numpy as np
        with av.open(str(source_video)) as container:
            decoded_frame = next(container.decode(video=0), None)
            if decoded_frame is None:
                raise RuntimeError("优化后标注视频没有可解码的帧")
            frame_width, frame_height = decoded_frame.width, decoded_frame.height
        annotated_frames = {}
        for side in ("left", "right"):
            with np.load(output / "hand_processor" / f"hand_data_{side}.npz") as data:
                points = data["kpts_2d"]
                visible = (np.isfinite(points).all(axis=2) & (points[:, :, 0] > 0)
                           & (points[:, :, 0] < frame_width) & (points[:, :, 1] > 0)
                           & (points[:, :, 1] < frame_height))
                annotated_frames[side] = int((data["hand_detected"] & visible.any(axis=1)).sum())
        (output / "annotation_counts.json").write_text(json.dumps(annotated_frames, indent=2))
        if not any(annotated_frames.values()):
            print("提示：两只手都没有有效标注点，视频仍按官方行为输出", flush=True)
        shutil.copy2(source_video, final_video)
    marker.write_text(json.dumps({"stage": args.stage, "path": path_mode, "camera_source": camera_source,
                                  "optimized_video": str(final_video) if final_video else None,
                                  "optimization": optimization}, ensure_ascii=False, indent=2))
    report_progress("完成", message=str(final_video or output / "hand_processor"))
    print(f"实验完成，标注视频位于：{output / 'hand_processor'}")
    return 0


if __name__ == "__main__":
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("PYTHONPATH", "")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
