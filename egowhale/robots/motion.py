"""Shared named-motion serialization; called by nodes, never by asset adapters."""
import json

import numpy as np


def read_motion(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def frame_count(motion):
    return len(motion["joint_positions"] if "joint_positions" in motion else motion["qpos"])


def read_bases(path):
    return read_base_placement(path)[0]


def read_base_placement(path):
    data = json.loads(path.read_text())
    if "bases" in data:
        bases = {name: np.asarray(matrix, dtype=np.float64) for name, matrix in data["bases"].items()}
    else:
        bases = np.asarray(data["matrix"], dtype=np.float64)
    frame = data.get("base_frame", "camera")
    if frame not in ("world", "camera"):
        raise ValueError(f"Unknown base frame: {frame}")
    return bases, frame


def read_camera_to_world(src, start=0, stop=None):
    import h5py

    with h5py.File(src, "r") as handle:
        camera = np.asarray(handle["transforms/camera"][start:stop], dtype=np.float32)
    if camera.ndim != 3 or camera.shape[1:] != (4, 4) or not np.isfinite(camera).all():
        raise ValueError(f"Invalid camera-to-world transforms in {src}")
    return camera


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
