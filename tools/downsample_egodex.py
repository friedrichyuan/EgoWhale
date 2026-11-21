"""Create a separate EgoDex dataset retaining 60 percent of frames with synchronized subsampling."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import uuid
from pathlib import Path

import cv2
import h5py
import numpy as np


def mapping(count):
    # Keep ceil(3*T/5) frames, sampling source time at uniform 5/3-frame intervals.
    return np.arange((count * 3 + 4) // 5, dtype=np.int64) * 5 // 3


def retime_h5(src, dst, index, fps):
    with h5py.File(src) as inp, h5py.File(dst, "w") as out:
        for k, v in inp.attrs.items():
            out.attrs[k] = v

        def copy(name, obj):
            if isinstance(obj, h5py.Group):
                target = out.require_group(name)
            else:
                values = obj[...]
                if name.startswith(("transforms/", "confidences/")):
                    if len(values) != len(inp["transforms/camera"]):
                        raise ValueError(f"unaligned {name}")
                    values = values[index]
                elif name != "camera/intrinsic":
                    raise ValueError(f"unsupported dataset {name}; explicit time-axis handling required")
                target = out.create_dataset(name, data=values, compression="gzip" if values.ndim else None)
            for k, v in obj.attrs.items():
                target.attrs[k] = v

        inp.visititems(copy)
        out.attrs["egowhale_frame_keep_ratio"] = 0.6
        out.attrs["egowhale_fps"] = fps
        out.attrs["egowhale_source"] = str(src)


def convert(src, out, crf, threads):
    video = src.with_suffix(".mp4")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot read {video}")
    with h5py.File(src) as f:
        count = len(f["transforms/camera"])
    if count < 1:
        raise ValueError("empty episode")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid source FPS: {fps}")
    output_fps = fps * 0.6
    width, height = int(cap.get(3)), int(cap.get(4))
    index = mapping(count)
    target = out / "test" / src.parent.name / src.name
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target.with_suffix(".downsample.json")
    signature = dict(
        source=str(src),
        h5_size=src.stat().st_size,
        h5_mtime_ns=src.stat().st_mtime_ns,
        video_size=video.stat().st_size,
        video_mtime_ns=video.stat().st_mtime_ns,
        crf=crf,
        input_frames=count,
        output_frames=len(index),
        source_fps=fps,
        fps=output_fps,
        frame_keep_ratio=0.6,
    )
    if marker.exists():
        previous = json.loads(marker.read_text())
        assert previous == signature and target.is_file() and target.with_suffix(".mp4").is_file(), "resume mismatch"
        cap.release()
        return signature
    if target.exists() or target.with_suffix(".mp4").exists():
        raise FileExistsError(f"uncommitted outputs at {target}")
    temp = target.parent / ("." + src.stem + ".tmp_" + uuid.uuid4().hex)
    temp.mkdir()
    tmpvideo = temp / "video.mp4"
    ff = subprocess.Popen(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(output_fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            str(threads),
            "-crf",
            str(crf),
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmpvideo),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        position = 0
        for i in range(count):
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"video ends at {i}, expected {count}")
            if position < len(index) and index[position] == i:
                ff.stdin.write(frame.tobytes())
                position += 1
        if cap.read()[0]:
            raise ValueError("video longer than HDF5")
        ff.stdin.close()
        if ff.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    finally:
        cap.release()
        if ff.poll() is None:
            ff.terminate()
            ff.wait()
    tmph5 = temp / "poses.hdf5"
    retime_h5(src, tmph5, index, output_fps)
    # Decode all output frames and verify every time-aligned HDF5 array, not just container metadata.
    check = cv2.VideoCapture(str(tmpvideo))
    decoded = 0
    try:
        assert math.isclose(check.get(cv2.CAP_PROP_FPS), output_fps, abs_tol=1e-3)
        while check.read()[0]:
            decoded += 1
    finally:
        check.release()
    assert decoded == len(index), (decoded, len(index))
    with h5py.File(src) as inp, h5py.File(tmph5) as result:
        for group in ("transforms", "confidences"):
            # EgoDex confidence annotations are optional; preserve absence, never invent values.
            if group == "confidences" and group not in inp:
                assert group not in result, "unexpected synthesized confidences"
                continue
            for name in inp[group]:
                np.testing.assert_array_equal(result[group + "/" + name][...], inp[group + "/" + name][...][index])
        np.testing.assert_array_equal(result["camera/intrinsic"][...], inp["camera/intrinsic"][...])
    os.replace(tmph5, target)
    os.replace(tmpvideo, target.with_suffix(".mp4"))
    marker.write_text(json.dumps(signature, indent=2))
    temp.rmdir()
    return signature


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--crf", type=int, default=18)
    a = p.parse_args()
    if a.limit < 0 or a.threads < 1 or not 0 <= a.crf <= 51:
        p.error("invalid limit/threads/crf")
    sources = [Path(s.strip()).resolve() for s in a.episodes.read_text().splitlines() if s.strip()]
    if a.limit:
        sources = sources[: a.limit]
    keys = [(s.parent.name, s.name) for s in sources]
    if not sources or len(set(keys)) != len(keys):
        p.error("empty list or duplicate task/episode")
    a.out = a.out.resolve()
    if any(s.is_relative_to(a.out) for s in sources):
        p.error("output must be separate from source dataset")
    manifest = dict(
        sources=[str(s) for s in sources],
        frame_keep_ratio=0.6,
        fps_policy="source_fps_times_0.6",
        schema_version=2,
        crf=a.crf,
        method="synchronized_frame_subsampling",
        source_list_sha256=hashlib.sha256(a.episodes.read_bytes()).hexdigest(),
    )
    a.out.mkdir(parents=True, exist_ok=True)
    lock = (a.out / ".conversion.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error("another conversion is running in this output directory")
    frozen = a.out / "conversion.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == manifest, "conversion settings differ"
    else:
        frozen.write_text(json.dumps(manifest, indent=2))
    cv2.setNumThreads(1)
    for i, src in enumerate(sources):
        row = convert(src, a.out, a.crf, a.threads)
        print(json.dumps(dict(episode=i + 1, total=len(sources), **row)), flush=True)
    # Publish the ready-to-use list only when every episode has committed outputs.
    text = "".join(str(a.out / "test" / s.parent.name / s.name) + "\n" for s in sources)
    (a.out / "episodes.txt.tmp").write_text(text)
    os.replace(a.out / "episodes.txt.tmp", a.out / "episodes.txt")
    print(f"COMPLETE {len(sources)} episodes: {a.out / 'episodes.txt'}", flush=True)


if __name__ == "__main__":
    main()
