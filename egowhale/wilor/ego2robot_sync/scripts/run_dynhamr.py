#!/usr/bin/env python3
"""Run legacy DynHaMR/HMP in the current SAM3-compatible Python environment."""
import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def install_static_camera_intrinsics(camera_class):
    """Keep static extrinsics while using the explicitly supplied calibration.

    Upstream's static branch otherwise ignores cameras.npz and substitutes
    (width + height) / 2, which disagrees with DA3 and our output projection.
    """
    original = camera_class.load_data

    def load_data(camera):
        if not camera.is_static:
            return original(camera)
        import numpy as np
        import torch
        path = Path(camera.cam_dir) / "cameras.npz"
        with np.load(path) as data:
            width, height = int(data["width"]), int(data["height"])
            intrins = np.asarray(data["intrins"], dtype=np.float32)
        if min(width, height) <= 0 or intrins.ndim != 2 or intrins.shape[1] != 4:
            raise ValueError(f"Invalid camera calibration: {path}")
        selected = intrins[camera.sidx:camera.eidx].copy()
        if (selected.shape != (camera.seq_len, 4) or not np.isfinite(selected).all()
                or (selected[:, :2] <= 0).any()):
            raise ValueError(f"Camera calibration does not match optimized timeline: {path}")
        img_w, img_h = camera.img_size
        selected *= np.array([img_w / width, img_h / height, img_w / width, img_h / height])
        camera.intrins = torch.from_numpy(selected)
        camera.cam_R = torch.eye(3)[None].repeat(camera.seq_len, 1, 1)
        camera.cam_t = torch.zeros(camera.seq_len, 3)
        print(f"Static camera using supplied intrinsics: {selected[0].tolist()} from {path}", flush=True)

    camera_class.load_data = load_data


def install_hmp_fallback(fitting, status_path):
    """Let root/smooth errors propagate; only a failed HMP call is optional."""
    from ego2robot.utils.progress import report_progress
    original = fitting.run_prior
    status_path.parent.mkdir(parents=True, exist_ok=True)

    def optional_prior(cfg, dataset, out_dir, *args, **kwargs):
        # Reaching this call proves that both root and smooth returned normally.
        # Keep the exact output path from this run, rather than searching old runs.
        paths = sorted((Path(out_dir) / "smooth_fit").glob(f"{dataset.seq_name}_*_world_results.npz"))
        smooth = {"root_smooth_completed": True,
                  "smooth_result": str(paths[-1].resolve()) if paths else None}
        count = int(dataset.seq_len)
        status_path.with_name("smooth_completed.json").write_text(json.dumps({**smooth, "frames": count}))
        running = status_path.with_name("hmp_running.json")
        running.write_text(json.dumps({"stage": "hmp", **smooth}))
        try:
            if count != 128:
                raise ValueError(f"HMP skipped: input has {count} frames; requires exactly 128")
            return original(cfg, dataset, out_dir, *args, **kwargs)
        except Exception as error:
            status = {"stage": "hmp", "fallback": True, "succeeded": False,
                      **smooth, "skipped": count != 128,
                      "reason": f"{type(error).__name__}: {error}"}
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
            temporary.replace(status_path)
            report_progress("HMP 未完成，准备检查 smooth 结果", message=status["reason"])
            return None
        finally:
            running.unlink(missing_ok=True)

    fitting.run_prior = optional_prior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynhamr-root", type=Path, default=ROOT / "thirdparty/Dyn-HaMR")
    parser.add_argument("--check-import", action="store_true")
    parser.add_argument("--hmp-fallback-status", type=Path)
    args, overrides = parser.parse_known_args()
    source = args.dynhamr_root.resolve() / "dyn-hamr"
    if not (source / "run_opt.py").is_file():
        raise FileNotFoundError(source / "run_opt.py")
    for path in (ROOT, source / "HMP", source):
        sys.path.insert(0, str(path))
    from ego2robot.utils.legacy_mano import prepare_mano_loading
    prepare_mano_loading()
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.chdir(source)
    if args.check_import:
        importlib.import_module("run_opt")
        print(f"DynHaMR/HMP import OK: {sys.executable}", flush=True)
        return 0
    # HMP logs every loss evaluation at INFO; keep warnings and errors visible.
    from loguru import logger
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    if args.hmp_fallback_status:
        install_hmp_fallback(importlib.import_module("HMP.fitting"), args.hmp_fallback_status.resolve())
    install_static_camera_intrinsics(importlib.import_module("data.dataset").CameraData)
    sys.argv = [str(source / "run_opt.py"), *overrides]
    runpy.run_path(str(source / "run_opt.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
