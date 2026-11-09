"""Shared named-motion serialization; called by nodes, never by asset adapters."""
import json

import numpy as np


def read_motion(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def frame_count(motion):
    return len(motion["joint_positions"] if "joint_positions" in motion else motion["qpos"])


def read_bases(path):
    data = json.loads(path.read_text())
    if "bases" in data:
        return {name: np.asarray(matrix, dtype=np.float64) for name, matrix in data["bases"].items()}
    return np.asarray(data["matrix"], dtype=np.float64)


def prefix_length(dst):
    """Frames replacing IK frame zero; no approach means one unchanged IK frame."""
    from egowhale.paths import PREFIX

    path = dst / PREFIX
    if not path.is_file():
        return 1
    count = frame_count(read_motion(path))
    if count < 1:
        raise ValueError(f"Empty approach trajectory: {path}")
    return count
