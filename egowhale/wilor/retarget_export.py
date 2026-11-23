"""Export WiLoR predictions in the HDF5 layout read by EgoWhale Retarget."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np

RETARGET_HDF5 = "action/hands_wilor.hdf5"
RETARGET_VIDEO = "action/hands_wilor.mp4"
BATCH_RETARGET_HDF5 = "hands_wilor.hdf5"

# WiLoR uses OpenPose order: wrist, thumb, index, middle, ring, little.
JOINT_NAMES = (
    "Hand", "ThumbKnuckle", "ThumbIntermediateBase", "ThumbIntermediateTip", "ThumbTip",
    *(f"{finger}Finger{joint}" for finger in ("Index", "Middle", "Ring", "Little")
      for joint in ("Knuckle", "IntermediateBase", "IntermediateTip", "Tip")),
)


def export_retarget(
    dst: Path, result: dict, source: Path, signature: str, *, hdf5: str = RETARGET_HDF5,
) -> Path:
    """Write positional transforms with an identity camera for camera-frame joints."""
    source = source.resolve(strict=True)
    if source.suffix.lower() != ".mp4":
        raise ValueError(f"Retarget export needs the source MP4, got {source}")
    path = dst / hdf5
    video = path.with_suffix(".mp4")
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(result["joints_camera"], dtype=np.float32)
    valid = np.asarray(result["valid"], dtype=bool)
    frames = len(points)
    if points.shape != (frames, 2, len(JOINT_NAMES), 3) or valid.shape != (frames, 2) or not frames:
        raise ValueError("Invalid WiLoR joint timeline for Retarget")
    if not np.isfinite(points[valid]).all():
        raise ValueError("Valid WiLoR joints contain non-finite coordinates")
    intrinsic = np.asarray(result["intrinsic"])
    fps = float(result["fps"])
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all() or not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid WiLoR intrinsics or FPS for Retarget")

    if video.is_symlink():
        if video.resolve() != source:
            raise ValueError(f"Existing Retarget video points to a different MP4: {video}")
    else:
        video.symlink_to(os.path.relpath(source, video.parent))

    with TemporaryDirectory(prefix=".retarget-", dir=path.parent) as directory:
        temporary = Path(directory) / path.name
        with h5py.File(temporary, "w") as handle:
            handle.attrs["egowhale_source"] = str(source)
            handle.attrs["egowhale_fps"] = fps
            handle.attrs["wilor_signature"] = signature
            handle.attrs["coordinate_frame"] = "per-frame camera"
            handle.attrs["joint_rotation"] = "identity placeholder; translation contains the estimated joint position"
            handle.create_dataset("camera/intrinsic", data=intrinsic)

            # Retarget applies inverse(transforms/camera). An identity keeps these camera-frame
            # points in place without mixing DA3's scene scale with the hand's depth estimate.
            transforms = handle.create_group("transforms")
            pose = np.broadcast_to(np.eye(4, dtype=np.float32), (frames, 4, 4)).copy()
            transforms.create_dataset("camera", data=pose, compression="gzip", compression_opts=1)
            for i, side in enumerate(("left", "right")):
                for j, name in enumerate(JOINT_NAMES):
                    pose[:, :3, 3] = np.where(valid[:, i, None], points[:, i, j], np.nan)
                    transforms.create_dataset(side + name, data=pose, compression="gzip", compression_opts=1)

            wilor = handle.create_group("wilor")
            for field in ("valid", "frame_index", "timestamps", "keypoints_2d"):
                wilor.create_dataset(field, data=result[field], compression="gzip", compression_opts=1)
        temporary.replace(path)
    return path


def validate_retarget(dst: Path, signature: str, frames: int, *, hdf5: str = RETARGET_HDF5) -> None:
    """Check that the published HDF5 and sibling video belong to this prediction."""
    path = dst / hdf5
    if not path.is_file() or not path.with_suffix(".mp4").is_file():
        raise ValueError(f"Incomplete WiLoR Retarget export: {path}")
    with h5py.File(path, "r") as handle:
        if (handle.attrs.get("wilor_signature") != signature
                or handle["transforms/camera"].shape != (frames, 4, 4)
                or handle["wilor/valid"].shape != (frames, 2)):
            raise ValueError(f"WiLoR Retarget export does not match its NPZ: {path}")


def export_existing(job: Path) -> Path:
    """Add a Retarget export to a completed WiLoR output without re-running inference."""
    action = job / "action"
    metadata = json.loads((action / "hands_wilor.json").read_text())
    # Re-export in place when a previous run already published a pair.
    hdf5 = metadata.get("retarget_hdf5") or RETARGET_HDF5
    source = Path(metadata["input"]["path"])
    stat = source.stat()
    if (stat.st_size, stat.st_mtime_ns) != (metadata["input"]["size"], metadata["input"]["mtime_ns"]):
        raise ValueError(f"The source MP4 changed since WiLoR ran: {source}")
    with np.load(action / "hands_wilor.npz", allow_pickle=False) as result:
        sig = metadata["signature"]
        if str(result["signature"]) != sig or len(result["valid"]) != metadata["frames"]:
            raise ValueError(f"WiLoR NPZ/metadata mismatch: {job}")
        return export_retarget(job, result, source, sig, hdf5=hdf5)
