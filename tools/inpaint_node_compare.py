"""Compare the original and current Inpaint backends on a fixed episode manifest."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

METHODS = {"propainter_v0": 0, "propainter_v1": 1, "sttn": 2}
PHASES = ("fps", "psnr", "lpips")
MAX_PIECE = 300
BATCH = 8
OUT_SIZE = (960, 540)
GUARD = 25


def write_json(path, data):
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def digest(path):
    import hashlib

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_inputs(args):
    import cv2
    import numpy as np

    from egowhale.pieces import piece_dir, split

    cv2.setNumThreads(1)
    selected = json.loads(args.selection.read_text())
    manifest = []
    for row in selected["episodes"]:
        source, video = Path(row["hdf5"]), Path(row["video"])
        frames = row["frames"]
        capture = cv2.VideoCapture(str(video))
        shape = [int(capture.get(key)) for key in (
            cv2.CAP_PROP_FRAME_COUNT, cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
        )]
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if not source.is_file() or shape != [frames, 1920, 1080] or fps != row["fps"]:
            raise ValueError(f"Source header mismatch: {video}: {shape}, {fps}")
        mask_root = args.masks / row["task"] / row["episode"]
        plan = json.loads((mask_root / "pieces.json").read_text())
        spans = plan["pieces"]
        if plan["frames"] != frames or spans[0][0] != 0 or spans[-1][1] != frames:
            raise ValueError(f"Mask plan mismatch: {mask_root}")
        parts = []
        for index, (start, stop) in enumerate(spans):
            path = piece_dir(mask_root, index, len(spans)) / "visual/masks.npz"
            with np.load(path) as saved:
                if saved["shape"].tolist() != [stop - start, 1080, 1920]:
                    raise ValueError(f"Mask shape mismatch: {path}")
            for left, right in split(stop - start, MAX_PIECE):
                parts.append(dict(mask=str(path), start=start + left, stop=start + right, mask_start=left))
        if sum(p["stop"] - p["start"] for p in parts) != frames:
            raise ValueError(f"Nonmatching frame count: {source}")
        manifest.append(dict(row, parts=parts))
    keys = [(row["task"], row["episode"]) for row in manifest]
    if len(keys) != len(set(keys)) or len(keys) < 2:
        raise ValueError("The manifest must have unique episodes and at least two videos")
    manifest_path = args.output / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Use a new output directory for a different set of inputs: {manifest_path}")

    native_weights = args.native / "third_party/ProPainter/weights"
    current_weights = args.thirdparty / "propainter/weights"
    hashes = {}
    for name in ("raft-things.pth", "recurrent_flow_completion.pth", "ProPainter.pth"):
        values = [digest(folder / name) for folder in (native_weights, current_weights)]
        if values[0] != values[1]:
            raise ValueError(f"ProPainter checkpoint mismatch: {name}")
        hashes[name] = values[0]
    weight = args.thirdparty / "sttn/checkpoints/sttn.pth"
    hashes[weight.name] = digest(weight)
    code = [
        ROOT / "egowhale" / name for name in ("visual/inpaint.py", "visual/paint.py", "visual/sttn.py", "media.py")
    ] + [args.native / "ego2robot/processors/handinpaint_processor.py", args.native_config, Path(__file__)]
    write_json(args.output / "metadata.json", {
        "selection": str(args.selection), "mask_root": str(args.masks),
        "selection_sha256": digest(args.selection), "code_sha256": {str(path): digest(path) for path in code},
        "native": str(args.native), "native_config": str(args.native_config),
        "thirdparty": str(args.thirdparty), "gpu_mapping": METHODS, "checkpoints_sha256": hashes,
        "episodes": len(manifest), "frames": sum(row["frames"] for row in manifest),
        "pieces": sum(len(row["parts"]) for row in manifest), "max_piece_frames": MAX_PIECE,
        "output_size": OUT_SIZE, "fps_warmup": "First piece of episode 1, repeated in formal measurement",
        "fps_timing": "Per GPU and configuration, divide verified frames by one continuous monotonic interval from the start of the first formal episode through the last validated output. Includes mask staging, decoding, inference, output writing and time between episodes. Model loading and warmup excluded.",
        "quality_mask": "Cyclic next episode's person mask, indexed modulo donor length, unioned with the real mask. Evaluate the borrowed hole at half resolution farther than 25 source pixels from the real mask.",
        "psnr_aggregation": "Global sum of squared RGB error divided by the total number of evaluated RGB channels before conversion to dB.",
        "lpips_aggregation": "Mean AlexNet LPIPS (v0.1) over scored frames. Outside the score mask the prediction is replaced by the reference. Crops enclose each 8-frame batch's score mask, padded to at least 32 pixels per side.",
        "quality_output": "Score the common 960x540 RGB output before codec encoding. PSNR saves only scored RGB pixels for the separate LPIPS stage.",
    })
    write_json(manifest_path, manifest)
    print(f"Ready: {len(manifest)} episodes, {sum(row['frames'] for row in manifest)} frames, {sum(len(row['parts']) for row in manifest)} pieces", flush=True)


def make_model(args, method, folder):
    if method == "propainter_v0":
        from omegaconf import OmegaConf

        sys.path.insert(0, str(args.native))
        os.environ["PROPAINTER_ROOT"] = str(args.native / "third_party/ProPainter")
        from ego2robot.processors import handinpaint_processor

        cfg = OmegaConf.load(args.native_config)
        cfg.data_root_dir = cfg.processed_data_root_dir = str(folder / "native")
        cfg.demo_name, cfg.skip_existing = "demos", False
        folder.joinpath("native/demos").mkdir(parents=True, exist_ok=True)
        return handinpaint_processor.HandInpaintProcessorPlus(cfg), handinpaint_processor

    from egowhale import paths, thirdparty

    # Use the shared checkpoint installation with this checkout's algorithms.
    paths.THIRDPARTY = thirdparty.THIRDPARTY = args.thirdparty
    from egowhale.visual.inpaint import Inpaint

    node = Inpaint(model="propainter" if method == "propainter_v1" else "sttn")
    node.load()
    return node, None


def load_part(part):
    from egowhale.media import load_masks

    length = part["stop"] - part["start"]
    offset = part["mask_start"]
    return load_masks(Path(part["mask"]))[offset : offset + length]


def donor_frames(donor, start, stop):
    import numpy as np

    indices = np.arange(start, stop) % donor["frames"]
    selected = np.empty((len(indices), 1080, 1920), dtype=bool)
    for part in donor["parts"]:
        take = (indices >= part["start"]) & (indices < part["stop"])
        if take.any():
            selected[take] = load_part(part)[indices[take] - part["start"]]
    return selected


def quality_masks(masks, borrowed):
    import cv2
    import numpy as np

    hole = masks | borrowed
    region = np.empty((len(masks), OUT_SIZE[1], OUT_SIZE[0]), dtype=bool)
    for i, (real, artificial) in enumerate(zip(masks, borrowed)):
        real = cv2.resize(real.astype("uint8"), OUT_SIZE, interpolation=cv2.INTER_NEAREST)
        artificial = cv2.resize(artificial.astype("uint8"), OUT_SIZE, interpolation=cv2.INTER_NEAREST)
        distance = cv2.distanceTransform(1 - real, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        region[i] = artificial.astype(bool) & (distance > GUARD / 2)
    return hole, region


def reference_blocks(row, part, size=BATCH):
    from egowhale.media import read_chunks, resize_frames

    for _, block in read_chunks(Path(row["video"]), part["start"], part["stop"], size):
        yield resize_frames(block, OUT_SIZE)


class QualityWriter:
    def __init__(self, row, part, region, cache):
        self.references = iter(reference_blocks(row, part))
        self.region = region
        self.cache = cache
        self.position = self.pixels = self.sq = 0

    def write(self, images):
        import numpy as np

        if len(images) % BATCH and self.position + len(images) != len(self.region):
            raise ValueError("Unexpected output batch boundary")
        for start in range(0, len(images), BATCH):
            out = np.asarray(images[start : start + BATCH])
            ref = next(self.references)
            if out.shape != ref.shape or out.dtype != np.uint8:
                raise ValueError(f"Bad output: {out.shape} {out.dtype}, expected {ref.shape} uint8")
            region = self.region[self.position : self.position + len(out)]
            pixels = out[region]
            error = pixels.astype(np.int32) - ref[region]
            self.sq += int(np.square(error).sum(dtype=np.int64))
            self.pixels += len(pixels)
            self.position += len(out)
            self.cache.write(pixels.tobytes())

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def finish(self):
        if self.position != len(self.region) or next(self.references, None) is not None:
            raise ValueError("The evaluated output does not have the expected frame count")
        return {"sse": self.sq, "scored_pixels": self.pixels,
                "scored_frames": int(self.region.any(axis=(1, 2)).sum())}


def render_native(node, module, row, part, masks, folder, writer):
    import h5py
    import numpy as np

    from egowhale.media import read_rgb

    folder.mkdir(parents=True, exist_ok=True)
    paths = node.get_paths(str(folder))
    video, start, stop = Path(row["video"]), part["start"], part["stop"]
    source = Path(paths.video_rgb_imgs)
    if not source.is_symlink():
        source.symlink_to(video)
    with h5py.File(row["hdf5"]) as data:
        k = data["camera/intrinsic"][:]
    write_json(Path(paths.cam_intrinsics), dict(
        fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]),
        new_w=1920, new_h=1080, orig_w=1920, orig_h=1080, v_fov=60.0, fps=row["fps"],
    ))
    fd, name = tempfile.mkstemp(prefix="inpaint_v0_", suffix=".npy")
    os.close(fd)
    mask_path = Path(paths.masks_arm)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask_path.unlink(missing_ok=True)
    try:
        np.save(name, masks)
        mask_path.symlink_to(name)
        # The original reader decodes whole episodes. For shared pieces, decode the same frame span as v1 and STTN.
        read = module.media.read_video if start == 0 and stop == row["frames"] else lambda _: read_rgb(video, start, stop)[0]
        native_write = module.media.write_video if writer is None else lambda _path, frames, **_: writer.write(frames)
        video_api = SimpleNamespace(read_video=read, write_video=native_write)

        def run_command(command, *args, **kwargs):
            if command[0] == "ffmpeg" and str(paths.video_human_inpaint) in command:
                return subprocess.CompletedProcess(command, 0)
            return original_run(command, *args, **kwargs)

        original_run = subprocess.run
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(module, "media", video_api))
            if writer is not None:
                stack.enter_context(patch.object(subprocess, "run", run_command))
            before = time.perf_counter()
            node.process_one_demo(folder.name)
            seconds = time.perf_counter() - before
        output = Path(paths.video_human_inpaint)
        if writer is None and not output.with_suffix(".mp4").is_file():
            raise RuntimeError(f"Native preview output missing: {output.with_suffix('.mp4')}")
        return seconds, output
    finally:
        mask_path.unlink(missing_ok=True)
        Path(name).unlink(missing_ok=True)


def render_current(node, row, part, masks, folder, writer):
    from egowhale import media
    from egowhale.visual import paint, sttn

    video = Path(row["video"])
    start, stop = part["start"], part["stop"]
    output = folder / "inpaint.mp4"
    prepare = paint.prepare_crop if node.model == "propainter" else sttn.prepare_frames
    capture = contextlib.nullcontext() if writer is None else patch.object(media, "RgbWriter", lambda *_: writer)
    before = time.perf_counter()
    clip = prepare(video, start, stop, masks)
    clip.update(dst=str(output), source=str(video), span=(start, stop))
    painted, = node._paint(node.held, [clip])
    with capture:
        node.store(node._save(clip, painted))
    return time.perf_counter() - before, output


def lpips_score(model, row, part, region, cache):
    import numpy as np
    import torch

    def limits(start, stop, extent):
        left = min(max(0, start - max(0, 32 - (stop - start)) // 2), extent - 32)
        return left, min(extent, max(stop, left + 32))

    position = count = 0
    total = 0.0
    with cache.open("rb") as pixels, torch.inference_mode():
        for ref in reference_blocks(row, part):
            mask = region[position : position + len(ref)]
            position += len(ref)
            valid = mask.any(axis=(1, 2))
            if not valid.any():
                continue
            mask, ref = mask[valid], ref[valid]
            n = int(mask.sum())
            prediction = np.frombuffer(pixels.read(n * 3), dtype=np.uint8).reshape(n, 3)
            ys, xs = np.nonzero(mask.any(0))
            y0, y1 = limits(ys.min(), ys.max() + 1, ref.shape[1])
            x0, x1 = limits(xs.min(), xs.max() + 1, ref.shape[2])
            ref = np.ascontiguousarray(ref[:, y0:y1, x0:x1])
            mixed = ref.copy()
            mixed[mask[:, y0:y1, x0:x1]] = prediction
            first = torch.from_numpy(mixed).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            second = torch.from_numpy(ref).cuda().permute(0, 3, 1, 2).float() / 127.5 - 1
            total += float(model(first, second).sum())
            count += len(ref)
        if pixels.read(1) or position != len(region):
            raise ValueError(f"LPIPS cache length mismatch: {cache}")
    if not math.isfinite(total):
        raise ValueError(f"Non-finite LPIPS: {row['video']}")
    return {"lpips_sum": total, "scored_frames": count}


def worker(args, phase, method):
    folder = args.output / method
    folder.mkdir(exist_ok=True)
    with (folder / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        measure_phase(args, phase, method, folder)


def measure_phase(args, phase, method, folder):
    import cv2
    import numpy as np
    import torch

    gpu = METHODS[method]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu) or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Worker {method} requires only physical GPU {gpu} to be visible")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    rows = json.loads((args.output / "manifest.json").read_text())
    metrics = folder / f"{phase}.jsonl"
    timing = folder / "fps_timing.json"
    done = {}
    if metrics.is_file():
        done = {item["index"]: item for item in map(json.loads, metrics.read_text().splitlines())}
    if phase == "fps" and done and (len(done) != (1 if args.smoke else len(rows)) or not timing.is_file()):
        raise RuntimeError(f"Continuous FPS timing requires a fresh pass. Archive {metrics} before retrying.")
    if len(done) == (1 if args.smoke else len(rows)):
        return
    started = time.perf_counter()
    if phase == "lpips":
        import lpips

        model = lpips.LPIPS(net="alex", version="0.1", verbose=False).cuda().eval()
        native = None
    else:
        model, native = make_model(args, method, folder)
    print(f"{method} GPU {gpu} {phase}: model ready in {time.perf_counter()-started:.1f}s", flush=True)

    def render(row, part, masks, number, writer=None):
        if native is not None:
            return render_native(model, native, row, part, masks, folder / "native/demos" / str(number), writer)
        return render_current(model, row, part, masks, folder / "videos" / str(number), writer)

    if phase == "fps":
        warm = rows[0]
        part = warm["parts"][0]
        _, path = render(warm, part, load_part(part), "warmup")
        check_output(path, part)
        print(f"{method} GPU {gpu} warmup done", flush=True)
    source = itertools.islice(rows, 1) if args.smoke else rows
    fps_start = time.perf_counter() if phase == "fps" else None
    successful_frames = 0
    for index, row in enumerate(source):
        if index in done:
            continue
        result = dict(index=index, episode=f"{row['task']}/{row['episode']}", frames=row["frames"], parts=[])
        started = time.perf_counter()
        for part_index, part in enumerate(row["parts"]):
            number = index * 10000 + part_index
            cache = folder / "quality" / f"{number}.rgb"
            if phase == "fps":
                seconds, path = render(row, part, load_part(part), number)
                check_output(path, part)
                output = dict(seconds=seconds, video=str(path))
            elif phase == "psnr":
                donor = rows[(index + 1) % len(rows)]
                masks, region = quality_masks(load_part(part), donor_frames(donor, part["start"], part["stop"]))
                cache.parent.mkdir(exist_ok=True)
                temp = cache.with_suffix(".partial")
                with temp.open("wb") as data:
                    writer = QualityWriter(row, part, region, data)
                    render(row, part, masks, number, writer)
                    output = writer.finish()
                temp.replace(cache)
                np.savez_compressed(cache.with_suffix(".region.npz"), packed=np.packbits(region, axis=-1))
            else:
                with np.load(cache.with_suffix(".region.npz")) as saved:
                    region = np.unpackbits(saved["packed"], axis=-1, count=OUT_SIZE[0]).astype(bool)
                output = lpips_score(model, row, part, region, cache)
            result["parts"].append(dict(start=part["start"], stop=part["stop"], **output))
            if len(row["parts"]) > 1:
                print(f"{phase} {method} {index + 1}/{len(rows)} {result['episode']} part {part_index+1}/{len(row['parts'])}", flush=True)
        episode_finished = time.perf_counter()
        result["episode_wall_s"] = episode_finished - started
        successful_frames += result["frames"]
        with metrics.open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        print(f"{phase} {method} GPU {gpu} {index + 1}/{len(rows)} {result['episode']} ({result['episode_wall_s']:.1f}s)", flush=True)
    if fps_start is not None:
        elapsed = episode_finished - fps_start
        write_json(timing, {"elapsed_wall_s": elapsed, "successful_frames": successful_frames})


def check_output(path, part):
    from egowhale.media import video_shape

    frames, height, width, _ = video_shape(path)
    if (frames, width, height) != (part["stop"] - part["start"], *OUT_SIZE):
        raise ValueError(f"Unexpected output video dimensions: {path}, {frames} frames, {width}x{height}")


def summarize(args, phase):
    data = {}
    expected = json.loads((args.output / "manifest.json").read_text())
    columns = {key: {} for key in METHODS}
    scored = None
    metrics_csv = []
    for method in METHODS:
        rows = list(map(json.loads, (args.output / method / f"{phase}.jsonl").read_text().splitlines()))
        if (len(rows) != len(expected)
                or [row["index"] for row in rows] != list(range(len(expected)))
                or [r["frames"] for r in rows] != [r["frames"] for r in expected]):
            raise RuntimeError(f"Incomplete result for {phase} {method}")
        value = {"episodes": len(rows), "frames": sum(item["frames"] for item in rows)}
        parts = [part for item in rows for part in item["parts"]]
        if phase == "fps":
            timing = json.loads((args.output / method / "fps_timing.json").read_text())
            elapsed = timing["elapsed_wall_s"]
            episode_wall_s = sum(item["episode_wall_s"] for item in rows)
            value["node_seconds"] = sum(part["seconds"] for part in parts)
            if value["frames"] != timing["successful_frames"] or not (
                0 < value["node_seconds"] <= episode_wall_s <= elapsed < math.inf
            ):
                raise RuntimeError(f"Inconsistent continuous FPS timing for {method}: {timing}")
            value["elapsed_wall_s"] = elapsed
            value["fps"] = value["frames"] / elapsed
            value["node_fps"] = value["frames"] / value["node_seconds"]
            value["episode_wall_fps"] = value["frames"] / episode_wall_s
        else:
            good = [part["scored_frames"] for part in parts]
            coverage = [(part["scored_frames"], part.get("scored_pixels")) for part in parts]
            if not sum(good) or (scored is not None and scored != coverage):
                raise RuntimeError("Methods must score the exact same frames and pixels")
            scored = coverage
            value["scored_frames"] = sum(good)
            if phase == "psnr":
                value["scored_pixels"] = sum(part["scored_pixels"] for part in parts)
                value["sse"] = sum(part["sse"] for part in parts)
                value["psnr"] = 10 * math.log10(255 ** 2 * 3 * value["scored_pixels"] / max(value["sse"], 1))
            else:
                psnr_rows = list(map(json.loads, (args.output / method / "psnr.jsonl").read_text().splitlines()))
                if good != [part["scored_frames"] for row in psnr_rows for part in row["parts"]]:
                    raise RuntimeError(f"LPIPS and PSNR scored frames differ for {method}")
                value["lpips"] = sum(part["lpips_sum"] for part in parts) / value["scored_frames"]
        data[method] = value
        for item in rows:
            entry = dict(method=method, gpu=METHODS[method], episode=item["episode"], frames=item["frames"])
            if phase == "fps":
                entry[phase] = item["frames"] / item["episode_wall_s"]
            else:
                entry["scored_frames"] = sum(p["scored_frames"] for p in item["parts"])
                if phase == "psnr":
                    pixels = sum(p["scored_pixels"] for p in item["parts"])
                    sse = sum(p["sse"] for p in item["parts"])
                    entry[phase] = 10 * math.log10(255 ** 2 * 3 * pixels / max(sse, 1)) if pixels else None
                else:
                    total = sum(p["lpips_sum"] for p in item["parts"])
                    entry[phase] = total / entry["scored_frames"] if entry["scored_frames"] else None
            metrics_csv.append(entry)
    with (args.output / f"{phase}_episodes.csv").open("w", newline="") as stream:
        csv_writer = csv.DictWriter(stream, fieldnames=list(metrics_csv[0]))
        csv_writer.writeheader()
        csv_writer.writerows(metrics_csv)
    write_json(args.output / f"{phase}_summary.json", data)
    for metric in PHASES:
        path = args.output / f"{metric}_summary.json"
        if path.is_file():
            for method, row in json.loads(path.read_text()).items():
                columns[method][metric] = row[metric]
    write_json(args.output / "results.json", columns)
    lines = ["| Model | GPU | FPS ↑ | PSNR (dB) ↑ | LPIPS ↓ |", "|---|---:|---:|---:|---:|"]
    for method, values in columns.items():
        metrics = [f"{values[metric]:.4f}" if metric in values else "pending" for metric in PHASES]
        lines.append(f"| {method} | {METHODS[method]} | " + " | ".join(metrics) + " |")
    (args.output / "results.md").write_text("\n".join(lines) + "\n")


def benchmark_command(args, *options):
    command = [sys.executable, "-u", str(Path(__file__).resolve())]
    for name in ("selection", "masks", "native", "native_config", "thirdparty", "output", "deps"):
        command.extend((f"--{name.replace('_', '-')}", str(getattr(args, name))))
    if args.smoke:
        command.append("--smoke")
    return [*command, *options]


def verify_lpips(args):
    alexnet = args.deps / "torch/hub/checkpoints/alexnet-owt-7be5be79.pth"
    download_hash = digest(alexnet)
    if not download_hash.startswith("7be5be79"):
        raise ValueError(f"Incomplete or invalid AlexNet checkpoint: {alexnet}")
    write_json(args.output / "lpips_checkpoint.json", {"sha256": download_hash, "package": "lpips 0.1.4"})


def run_method(args, method):
    gpu = METHODS[method]
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < len(METHODS):
        raise RuntimeError("Three workers need at least three logical CPUs")
    cpus_per_worker = len(allowed) // len(METHODS)
    cpu = allowed[gpu * cpus_per_worker : (gpu + 1) * cpus_per_worker]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCH_HOME=str(args.deps / "torch"), MPLCONFIGDIR=str(args.output / "matplotlib"),
               PYTHONPATH=os.pathsep.join(filter(None, (str(args.deps), os.environ.get("PYTHONPATH")))))
    folder = args.output / method
    folder.mkdir(exist_ok=True)
    state_path = folder / "state.json"
    with (folder / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = []
        for phase in PHASES:
            state = {"status": "running", "phase": phase, "pid": os.getpid(), "gpu": gpu, "completed": completed}
            write_json(state_path, state)
            print(f"START {method} {phase} GPU {gpu}", flush=True)
            try:
                if phase == "lpips":
                    verify_lpips(args)
                command = ["taskset", "-c", ",".join(map(str, cpu)),
                           *benchmark_command(args, "--worker", phase, method)]
                with (args.output / f"{phase}_{method}.log").open("a") as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            except BaseException as error:
                write_json(state_path, dict(state, status="failed", error=str(error)))
                raise
            completed.append(phase)
            print(f"DONE {method} {phase}", flush=True)
        write_json(state_path, {"status": "complete", "gpu": gpu, "completed": completed})


def update_summaries(args, published):
    state_paths = [args.output / method / "state.json" for method in METHODS]
    if not all(path.is_file() for path in state_paths):
        return
    completed = [json.loads(path.read_text()).get("completed", []) for path in state_paths]
    for phase in PHASES:
        if phase not in published and all(phase in phases for phases in completed):
            summarize(args, phase)
            published.add(phase)
            print(f"DONE {phase}", flush=True)


def run(args):
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (args.output / "controller.pid").write_text(f"{os.getpid()}\n")
        write_json(args.output / "state.json", {"status": "running", "pid": os.getpid()})
        processes = {}
        try:
            with contextlib.ExitStack() as logs:
                for method in METHODS:
                    log = logs.enter_context((args.output / f"pipeline_{method}.log").open("a"))
                    processes[method] = subprocess.Popen(
                        benchmark_command(args, "--method", method), stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                published = set()
                while True:
                    errors = {method: p.poll() for method, p in processes.items()}
                    if not args.smoke:
                        update_summaries(args, published)
                    if all(code is not None for code in errors.values()):
                        break
                    time.sleep(5)
            if any(errors.values()):
                raise RuntimeError(f"Method pipeline exit codes: {errors}")
        except BaseException as error:
            write_json(args.output / "state.json", {"status": "failed", "error": str(error)})
            raise
        finally:
            for process in processes.values():
                if process.poll() is None:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGTERM)
        write_json(args.output / "state.json", {"status": "smoke_passed" if args.smoke else "complete"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("selection", "masks", "native", "native-config", "thirdparty", "output", "deps"):
        parser.add_argument(f"--{flag}", type=lambda value: Path(value).resolve(), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--method", choices=METHODS, help="Advance one method on its assigned GPU")
    parser.add_argument("--worker", nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        inspect_inputs(args)
    elif args.worker:
        worker(args, *args.worker)
    elif args.method:
        run_method(args, args.method)
    else:
        run(args)


if __name__ == "__main__":
    main()
