"""Level 2: statistics of the retargeted action. Steps far outside the episode's own range, and sudden jumps
in position, velocity, or jerk, are flagged."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def trajectory(position, rotation, width):
    """State (both grippers' positions and widths) and action (their per-frame deltas, rotation as a rotvec)."""
    state = np.concatenate((position.reshape(len(position), -1), width), axis=1)
    pos_delta = np.diff(position.reshape(len(position), -1), axis=0)
    width_delta = np.diff(width, axis=0)
    rot_delta = np.full((len(position) - 1, 6), np.nan, dtype=np.float64)
    for side in range(2):
        good = np.isfinite(rotation[:-1, side]).all(axis=(1, 2)) & np.isfinite(rotation[1:, side]).all(axis=(1, 2))
        if not good.any():
            continue
        rot_delta[good, side * 3 : (side + 1) * 3] = (
            Rotation.from_matrix(rotation[:-1, side][good]).inv() * Rotation.from_matrix(rotation[1:, side][good])
        ).as_rotvec()
    action = np.concatenate((pos_delta, rot_delta, width_delta), axis=1)
    return state, action


def outliers(action: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Frames whose action leaves the episode's 1-99% band widened by three times its span."""
    flagged = np.zeros(len(valid), dtype=bool)
    usable = valid[:-1] & valid[1:] & np.isfinite(action).all(axis=1)
    if int(usable.sum()) < 4:
        return flagged
    low, high = _band(action[usable])
    outside = usable & np.any((action < low) | (action > high), axis=1)
    flagged[1:] = outside
    return flagged


def sudden(series: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Frames where the first, second, or third difference jumps past the same band."""
    flagged = np.zeros(len(series), dtype=bool)
    kernels = (
        np.array([-1.0, 1.0]),
        np.array([1.0, -2.0, 1.0]),
        np.array([-1.0, 3.0, -3.0, 1.0]),
    )
    finite = valid & np.isfinite(series).all(axis=1)
    for kernel in kernels:
        width = len(kernel)
        rows, ends = [], []
        for end in range(width - 1, len(series)):
            if not finite[end - width + 1 : end + 1].all():
                continue
            rows.append(series[end - width + 1 : end + 1].T @ kernel)
            ends.append(end)
        if len(rows) < 4:
            continue
        values = np.abs(np.stack(rows))
        limit = _upper(values)
        over = np.any(values > limit, axis=1)
        for end, hit in zip(ends, over):
            flagged[end] |= bool(hit)
    return flagged


def _band(values: np.ndarray):
    low = np.quantile(values, 0.01, axis=0)
    high = np.quantile(values, 0.99, axis=0)
    span = high - low
    return low - 3.0 * span, high + 3.0 * span


def _upper(values: np.ndarray) -> np.ndarray:
    _low, high = _band(values)
    return high
