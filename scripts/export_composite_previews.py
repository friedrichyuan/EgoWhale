#!/usr/bin/env python3
"""Export H.264 preview copies as OUT/task/episode.mp4."""
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path),
    ]))["streams"][0]


def convert(source, target):
    original = probe(source)
    expected = {**original, "codec_name": "h264", "pix_fmt": "yuv420p"}
    if target.exists():
        if probe(target) != expected:
            raise ValueError(f"Existing preview has different video metadata: {target}")
        return "exists"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".preview-") as staging:
        preview = Path(staging) / "video.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(source),
            "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-threads", "1", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(preview),
        ], check=True)
        if probe(preview) != expected:
            raise ValueError(f"Unexpected video metadata after conversion: {source}")
        os.link(preview, target)
    return "converted"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Composite output root")
    parser.add_argument("out", type=Path, help="Separate preview root")
    parser.add_argument("--jobs", type=int, default=4, help="Number of simultaneous CPU encodes")
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out.resolve()
    if args.jobs < 1 or out.is_relative_to(source) or source.is_relative_to(out):
        parser.error("Use a positive job count and separate source and output roots")
    files = sorted(source.glob("*/*/visual/composite.mp4"))
    if not files:
        parser.error(f"No task/episode/visual/composite.mp4 files under {source}")
    failures = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = {}
        for path in files:
            task, episode, _, _ = path.relative_to(source).parts
            target = out / task / f"{episode}.mp4"
            jobs[pool.submit(convert, path, target)] = target
        for index, job in enumerate(as_completed(jobs), 1):
            try:
                print(f"[{index}/{len(files)}] {job.result()} {jobs[job]}", flush=True)
            except Exception as exc:
                failures += 1
                print(f"[{index}/{len(files)}] failed {jobs[job]}: {exc}", flush=True)
    print(f"{len(files) - failures} ready, {failures} failed", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
