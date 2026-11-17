"""Conservative acceptance checks for a smooth result, without loading models."""
import numpy as np


def assess_smooth_result(baseline, candidate, frames, width, height, cfg):
    """Compare on the original timeline. These checks do not establish GT accuracy."""
    limits = {
        "min_retained_ratio": float(cfg.get("dynhamr_smooth_min_retained_ratio", 0.8)),
        "max_median_offset": float(cfg.get("dynhamr_smooth_max_median_offset", 0.05)),
        "max_p95_offset": float(cfg.get("dynhamr_smooth_max_p95_offset", 0.15)),
        "min_visible_ratio": float(cfg.get("dynhamr_smooth_min_visible_ratio", 0.5)),
    }
    if any(not np.isfinite(v) or not 0 <= v <= 1 for v in limits.values()):
        raise ValueError("smooth quality thresholds must be finite fractions in [0, 1]")
    diagonal = float(np.hypot(width, height))
    if frames <= 0 or min(width, height) <= 0:
        raise ValueError("invalid video dimensions")
    reasons, metrics = [], {}
    for side in ("left", "right"):
        old = baseline[side]
        before = np.asarray(old["hand_detected"], dtype=bool)
        previous_uv = np.asarray(old["kpts_2d"])
        if before.shape != (frames,) or previous_uv.shape != (frames, 21, 2):
            raise ValueError(f"invalid {side} baseline dimensions")
        stats = metrics[side] = {"baseline_frames": int(before.sum())}
        hand = candidate.get(side)
        if hand is None:
            if before.any():
                reasons.append(f"{side}: missing optimized hand")
            continue
        valid = np.asarray(hand["hand_detected"], dtype=bool)
        xyz, uv, rotation = (np.asarray(hand[key]) for key in ("kpts_3d", "kpts_2d", "eef_rot"))
        if (valid.shape != (frames,) or xyz.shape != (frames, 21, 3)
                or uv.shape != (frames, 21, 2) or rotation.shape != (frames, 3, 3)):
            reasons.append(f"{side}: invalid optimized array dimensions")
            continue
        stats["valid_frames"] = int(valid.sum())
        if valid.any():
            if not all(np.isfinite(a[valid]).all() for a in (xyz, uv, rotation)):
                reasons.append(f"{side}: non-finite optimized coordinates or rotations")
                continue
            if (xyz[valid, :, 2] <= 0).any():
                reasons.append(f"{side}: keypoints behind camera")
            visible = ((uv[valid, :, 0] >= 0) & (uv[valid, :, 0] < width)
                       & (uv[valid, :, 1] >= 0) & (uv[valid, :, 1] < height))
            stats["visible_point_ratio"] = float(visible.mean())
            if stats["visible_point_ratio"] < limits["min_visible_ratio"]:
                reasons.append(f"{side}: too many keypoints outside image")
        if not before.any():
            continue
        overlap = before & valid
        stats["retained_ratio"] = float(overlap.sum() / before.sum())
        if stats["retained_ratio"] < limits["min_retained_ratio"]:
            reasons.append(f"{side}: too few original detections retained")
        if not overlap.any():
            reasons.append(f"{side}: no comparable valid frames")
            continue
        offsets = np.linalg.norm(uv[overlap] - previous_uv[overlap], axis=-1) / diagonal
        if not np.isfinite(offsets).all():
            reasons.append(f"{side}: non-finite comparison points")
            continue
        stats["median_offset"] = float(np.median(offsets))
        stats["p95_offset"] = float(np.percentile(offsets, 95))
        if stats["median_offset"] > limits["max_median_offset"]:
            reasons.append(f"{side}: median 2D offset too large")
        if stats["p95_offset"] > limits["max_p95_offset"]:
            reasons.append(f"{side}: p95 2D offset too large")
    if not any(v["baseline_frames"] for v in metrics.values()):
        reasons.append("no baseline detections to validate")
    return {"accepted": not reasons, "reasons": reasons, "hands": metrics, "limits": limits}
