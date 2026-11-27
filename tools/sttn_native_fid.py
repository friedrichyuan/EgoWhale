"""Evaluate native 432x240 STTN predictions using the existing EgoDex score regions."""

from __future__ import annotations

import argparse
from itertools import batched
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate

from inpaint_fid import (
    DATASETS, DIMENSIONS, OUTPUTS, WEIGHT_SHA256, crop_slices, extract_features,
    fid_backend, read_jsonl, resize_groups,
)
from inpaint_node_compare import OUT_SIZE, digest, donor_frames, load_part, write_json
from egowhale.visual import sttn


CACHE = OUTPUTS / DATASETS["egodex"][0]
BASELINE = OUTPUTS / "inpaint_fid_node_bench100_20261126_170625/egodex.json"
NATIVE_SIZE = (432, 240)


def cohort(shards):
    manifest = json.loads((CACHE / "manifest.json").read_text())
    selected = CACHE.parents[1] / "datasets/node_bench100/episodes.txt"
    episode_paths = [Path(line) for line in selected.read_text().splitlines() if line]
    if len(manifest) != 100 or not all(
        path.samefile(row["hdf5"]) for path, row in zip(episode_paths, manifest, strict=True)
    ):
        raise ValueError("STTN cache does not match the node_bench100 cohort")
    scores = sorted(read_jsonl(CACHE / "sttn/psnr.jsonl"), key=lambda row: row["index"])
    if [row["index"] for row in scores] != list(range(len(manifest))):
        raise ValueError("Incomplete STTN score records")
    groups = [[] for _ in range(shards)]
    totals = [0] * shards
    for i in sorted(range(len(manifest)), key=lambda index: -manifest[index]["frames"]):
        target = totals.index(min(totals))
        groups[target].append(i)
        totals[target] += manifest[i]["frames"]
    return manifest, scores, [sorted(group) for group in groups]


def native_crop(region):
    # Project the original 960x540 FID crop onto the native image grid.
    return tuple(
        slice(bounds.start * target // source, -(-bounds.stop * target // source))
        for bounds, source, target in zip(crop_slices(region), OUT_SIZE[::-1], NATIVE_SIZE[::-1], strict=True)
    )


def as_rgb(frames):
    return frames.clamp(0, 1).mul(255).add(0.5).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def clip_crops(model, clip, region):
    predicted = sttn.predict_native(model, clip)
    if tuple(predicted.shape[-2:]) != NATIVE_SIZE[::-1]:
        raise ValueError(f"Unexpected native STTN dimensions {predicted.shape}")
    device = predicted.device
    frames = clip["frames"][0]
    for start in range(0, len(frames), 32):
        selected = slice(start, start + 32)
        old_region = region[selected]
        valid = old_region.any(axis=(1, 2))
        if not valid.any():
            continue
        reference = frames[selected].to(device).float().div_(255)
        reference = as_rgb(interpolate(reference, size=NATIVE_SIZE[::-1], mode="bilinear",
                                       antialias=True, align_corners=False))
        native_region = torch.from_numpy(old_region[:, None]).float()
        native_region = (interpolate(native_region, size=NATIVE_SIZE[::-1], mode="area") > 0)[:, 0].cpu().numpy()
        output = as_rgb(predicted[selected])
        for j in np.flatnonzero(valid):
            ys, xs = native_crop(old_region[j])
            score = native_region[j, ys, xs]
            if not score.any() or int(score.sum()) != int(native_region[j].sum()):
                raise ValueError("Native scoring crop lost selected pixels")
            ref = reference[j, ys, xs]
            mixed = ref.copy()
            mixed[score] = output[j, ys, xs][score]
            yield np.stack((ref, mixed))


def episode_crops(manifest, scores, model, index):
    row = manifest[index]
    donor = manifest[(index + 1) % len(manifest)]
    for part_index, (part, stats) in enumerate(zip(row["parts"], scores[index]["parts"], strict=True)):
        start, stop = part["start"], part["stop"]
        if (start, stop) != (stats["start"], stats["stop"]):
            raise ValueError(f"Mask span mismatch at episode {index} part {part_index}")
        saved = CACHE / "sttn/quality" / f"{index * 10000 + part_index}.region.npz"
        with np.load(saved) as array:
            region = np.unpackbits(array["packed"], axis=-1, count=OUT_SIZE[0]).astype(bool)
        if region.shape != (stop - start, OUT_SIZE[1], OUT_SIZE[0]):
            raise ValueError(f"Invalid shape in {saved}")
        if int(region.any(axis=(1, 2)).sum()) != stats["scored_frames"]:
            raise ValueError(f"Invalid scored frame count in {saved}")
        masks = load_part(part) | donor_frames(donor, start, stop)
        clip = sttn.prepare_frames(Path(row["video"]), start, stop, masks)
        del masks
        yield from clip_crops(model, clip, region)


def scored_frames(scores, indices):
    return sum(p["scored_frames"] for i in indices for p in scores[i]["parts"])


def compute(args, manifest, scores, indices):
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    device = torch.device("cuda:0")
    backend, _ = fid_backend(OUTPUTS / "inpaint_fid_deps")
    extractor = backend([backend.BLOCK_INDEX_BY_DIM[DIMENSIONS]], resize_input=False).to(device).eval()
    model = sttn.load_model(str(device))
    available = scored_frames(scores, indices)
    expected = scored_frames(scores, indices[:1]) if args.smoke else available
    features = np.empty((expected, 2, DIMENSIONS), dtype=np.float32)
    done = 0
    started = time.monotonic()
    with torch.inference_mode():
        for number, index in enumerate(indices[:1] if args.smoke else indices, 1):
            source = episode_crops(manifest, scores, model, index)
            for groups in batched(source, 24):
                batch = resize_groups(groups).flatten(0, 1)
                end = done + len(groups)
                features[done:end] = extract_features(extractor, batch, device).reshape(len(groups), 2, DIMENSIONS)
                done = end
            row = manifest[index]
            print(f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES')} episode {number}/{len(indices)} "
                  f"{row['task']}/{row['episode']} | features {done}/{expected}", flush=True)
    if done != expected or not np.isfinite(features).all():
        raise ValueError(f"Invalid feature count {done}/{expected}")
    if args.smoke:
        print("Native STTN FID feature smoke passed.", flush=True)
        return
    path = args.output / f"features_{args.shard}.npy"
    with path.with_suffix(".tmp").open("wb") as stream:
        np.save(stream, features, allow_pickle=False)
    path.with_suffix(".tmp").replace(path)
    write_json(args.output / f"shard_{args.shard}.json", {
        "indices": indices, "frames": done, "wall_seconds": time.monotonic() - started,
        "visible_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })


def summarize(output, manifest, scores, shards):
    baseline = json.loads(BASELINE.read_text())
    old = float(baseline["fid"]["sttn"])
    if baseline["source_manifests"]["sttn"]["sha256"] != digest(CACHE / "manifest.json"):
        raise ValueError("Baseline FID was calculated on a different manifest")
    arrays = []
    for shard, indices in enumerate(shards):
        rows = json.loads((output / f"shard_{shard}.json").read_text())
        data = np.load(output / f"features_{shard}.npy", allow_pickle=False, mmap_mode="r")
        if rows["indices"] != indices or data.shape != (scored_frames(scores, indices), 2, DIMENSIONS):
            raise ValueError(f"Incomplete STTN native FID shard {shard}")
        arrays.append(data)
    features = np.concatenate(arrays, axis=0)
    if len(features) != baseline["scored_frames"]:
        raise ValueError("Native and baseline FID do not use the same frames")
    _, distance = fid_backend(OUTPUTS / "inpaint_fid_deps")
    ref, out = features.transpose(1, 0, 2)
    fid = distance(ref.mean(0, dtype=np.float64), np.cov(ref, rowvar=False),
                   out.mean(0, dtype=np.float64), np.cov(out, rowvar=False))
    if not np.isfinite(fid) or fid < 0:
        raise ValueError(f"Invalid FID {fid}")
    report = {
        "metric": "FID on scored-region crops",
        "episodes": len(manifest), "scored_frames": len(features),
        "fid_432x240": float(fid), "fid_960x540": old, "native_minus_upsampled": float(fid - old),
        "native_size": NATIVE_SIZE, "reference_resize": "bilinear antialias, same as STTN input",
        "native_region": "Area projection of original score mask, includes any nonzero coverage",
        "crop": "Project original FID crop edges to native resolution",
        "baseline": str(BASELINE), "checkpoint_sha256": WEIGHT_SHA256,
        "sttn_checkpoint_sha256": digest(sttn._CKPT),
        "source_manifest_sha256": digest(CACHE / "manifest.json"),
    }
    write_json(output / "sttn_native_fid.json", report)
    (output / "results.md").write_text(
        "| Resolution | Scored frames | STTN FID ↓ |\n|---|---:|---:|\n"
        f"| 432×240 native | {len(features)} | {fid:.6f} |\n"
        f"| 960×540 upsampled | {len(features)} | {old:.6f} |\n"
    )
    print(f"STTN native FID {fid:.6f} vs upsampled {old:.6f} ({len(features)} frames)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summarize", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.shards <= 100 or not 0 <= args.shard < args.shards:
        parser.error("Invalid shard index or count")
    manifest, scores, shards = cohort(args.shards)
    if args.summarize:
        summarize(args.output, manifest, scores, shards)
    else:
        if not args.smoke:
            args.output.mkdir(parents=True, exist_ok=True)
        compute(args, manifest, scores, shards[args.shard])


if __name__ == "__main__":
    main()
