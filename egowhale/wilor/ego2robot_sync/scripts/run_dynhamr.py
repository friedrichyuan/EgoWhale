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


def install_hmp_fallback(fitting, status_path):
    """Let root/smooth errors propagate; only a failed HMP call is optional."""
    from ego2robot.utils.progress import report_progress
    original = fitting.run_prior
    status_path.parent.mkdir(parents=True, exist_ok=True)

    def optional_prior(*args, **kwargs):
        running = status_path.with_name("hmp_running.json")
        running.write_text(json.dumps({"stage": "hmp"}))
        try:
            return original(*args, **kwargs)
        except Exception as error:
            status = {"stage": "hmp", "fallback": True, "succeeded": False,
                      "result_source": "wilor_filtered", "reason": f"{type(error).__name__}: {error}"}
            status_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2))
            temporary.replace(status_path)
            report_progress("HMP 失败，回退 WiLoR 结果", message=status["reason"])
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
    sys.argv = [str(source / "run_opt.py"), *overrides]
    runpy.run_path(str(source / "run_opt.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
