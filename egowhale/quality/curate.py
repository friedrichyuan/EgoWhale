"""The curate stage: the three quality levels on one episode, written to action/quality.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from egowhale.media import frame_size, video_shape
from egowhale.paths import BASE, COMPOSITE, GRIPPER, IK, PREFIX, QUALITY
from egowhale.quality import actions, checks, vlm
from egowhale.step import Step

# An episode whose dropped share of frames is above this is not kept, and is not sent to the VLM.
DROP_RATIO = 0.60


class Curate(Step):
    name = "curate"
    needs = (GRIPPER, IK, BASE, COMPOSITE, PREFIX)
    makes = (QUALITY,)

    def run(self, src: Path, dst: Path) -> None:
        dst = Path(dst)
        with np.load(dst / GRIPPER) as data:
            position = np.asarray(data["position"], dtype=np.float64)
            rotation = np.asarray(data["rotation"], dtype=np.float64)
            width = np.asarray(data["width"], dtype=np.float64)
            valid = np.asarray(data["valid"], dtype=bool)
            fps = float(data["fps"])
            intrinsic = np.asarray(data["intrinsic"], dtype=np.float64).copy()
        with np.load(dst / IK) as data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
        base = np.asarray(json.loads((dst / BASE).read_text())["matrix"], dtype=np.float64)
        frames = min(len(position), len(qpos))
        position, rotation, width, valid, qpos = (
            position[:frames],
            rotation[:frames],
            width[:frames],
            valid[:frames],
            qpos[:frames],
        )
        _count, video_h, video_w, _video_fps = video_shape(dst / COMPOSITE)
        scale = video_h / frame_size(Path(src).with_suffix(".mp4"))[0]
        intrinsic[:2] *= scale
        signals = checks.frame_checks(qpos, base, intrinsic, position, valid, video_h, video_w)
        l1 = checks.erode(checks.passed(signals), fps)
        state, action = actions.trajectory(position, rotation, width)
        outlier = actions.outliers(action, l1)
        sudden = actions.sudden(state, l1)
        sudden[1:] |= actions.sudden(action, l1[:-1] & l1[1:])
        kept = l1 & ~outlier & ~sudden
        invalid_ratio = float((~kept).mean())
        keep = invalid_ratio <= DROP_RATIO
        description = vlm.description(Path(src))
        approach = len(np.load(dst / PREFIX)["qpos"]) - 1
        audit = vlm.audit(dst / COMPOSITE, approach, frames, fps, description) if keep else {"status": "skipped", "reason": "dropped before audit"}
        if audit.get("is_consistent") is False:
            keep = False
        report = {
            "keep": keep,
            "description": description,
            "invalid_ratio": invalid_ratio,
            "mean_ik_error_m": _finite_mean(signals["ik_error"]),
            "hand": _bits(signals["hand"]),
            "ik": _bits(signals["ik"]),
            "visible": _bits(signals["visible"]),
            "self_collision": _bits(signals["self_collision"]),
            "cross_arm": _bits(signals["cross_arm"]),
            "too_large": _bits(signals["coverage"]),
            "l1": _bits(l1),
            "outlier": _bits(outlier),
            "sudden": _bits(sudden),
            "valid": _bits(kept),
            "vlm": audit,
        }
        path = dst / QUALITY
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        print(f"  keep {int(kept.sum())}/{frames}  invalid {invalid_ratio:.1%}  vlm {audit['status']}")


def _finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return None
    return float(finite.mean())


def _bits(values: np.ndarray) -> list[int]:
    return [int(bool(value)) for value in values]
