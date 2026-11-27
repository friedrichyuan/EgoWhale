"""Score cached Inpaint predictions with Inception-v3 FID on the scored-region crops."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from functools import partial
from importlib.metadata import version
from itertools import batched
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from torch.nn.functional import interpolate

from inpaint_node_compare import METHODS, OUT_SIZE, ROOT, digest, reference_blocks, write_json


OUTPUTS = ROOT / "outputs"
DATASETS = {
    "egodex": ("inpaint_node_bench100_20261124",) * 3,
    "vitra": ("inpaint_vitra_20261125_QnQ04V",) * 3,
    "taco": ("inpaint_taco_propainter_20261125_iH5Evo",) * 2 + ("inpaint_taco_sttn_20261125_ZOhPUC",),
}
DIMENSIONS = 2048
CROP_PADDING = 16
REFERENCE_AND_METHODS = ("reference", *METHODS)
WEIGHT_FILENAME = "pt_inception-2015-12-05-6726825d.pth"
WEIGHT_SHA256 = "6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_cohort(roots):
    manifests = [root / "manifest.json" for root in roots]
    manifest = json.loads(manifests[0].read_text())
    for path in manifests[1:]:
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Different reference frame selection in {path}")
    models = [root / method for root, method in zip(roots, METHODS, strict=True)]
    metrics = []
    for model in models:
        records = sorted(read_jsonl(model / "psnr.jsonl"), key=lambda row: row["index"])
        if [r["index"] for r in records] != list(range(len(manifest))):
            raise ValueError(f"Incomplete or duplicate episode rows in {model}")
        metrics.append(records)
    expected = sum(p["scored_frames"] for row in metrics[0] for p in row["parts"])
    return manifest, models, metrics, expected


def region_for_part(cache_files, scored):
    masks = [path.with_suffix(".region.npz") for path in cache_files]
    with np.load(masks[0]) as archive:
        packed = archive["packed"]
    for path in masks[1:]:
        with np.load(path) as archive:
            if not np.array_equal(packed, archive["packed"]):
                raise ValueError(f"Score mask differs from {masks[0]} in {path}")
    region = np.unpackbits(packed, axis=-1, count=OUT_SIZE[0]).astype(bool)
    pixels = int(region.sum())
    frames = int(region.any(axis=(1, 2)).sum())
    for cache, entry in zip(cache_files, scored, strict=True):
        if (entry["scored_pixels"], entry["scored_frames"], cache.stat().st_size) != (pixels, frames, pixels * 3):
            raise ValueError(f"Mismatched prediction cache for {cache}")
    return region


def crop_slices(mask):
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))

    def bounds(points, extent):
        start = max(0, min(int(points[0]) - CROP_PADDING, extent - 32))
        stop = min(extent, max(int(points[-1]) + 1 + CROP_PADDING, start + 32))
        return slice(start, stop)

    return bounds(ys, mask.shape[0]), bounds(xs, mask.shape[1])


def crop_group(reference, mask, streams):
    crop = crop_slices(mask)
    region = mask[crop]
    ref = reference[crop]
    images = [ref]
    byte_count = int(np.count_nonzero(region)) * 3
    for stream in streams:
        data = stream.read(byte_count)
        if len(data) != byte_count:
            raise ValueError(f"Truncated RGB cache {stream.name}")
        mixed = ref.copy()
        mixed[region] = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3)
        images.append(mixed)
    return np.stack(images)


def cohort_crops(manifest, models, metrics):
    for index, row in enumerate(manifest):
        results = [method[index] for method in metrics]
        episode = f"{row['task']}/{row['episode']}"
        for result in results:
            if result["episode"] != episode or result["frames"] != row["frames"]:
                raise ValueError(f"Different source frames for {episode}")
        result_parts = [result["parts"] for result in results]
        for part_index, (part, *scored) in enumerate(zip(row["parts"], *result_parts, strict=True)):
            span = (part["start"], part["stop"])
            if any((entry["start"], entry["stop"]) != span for entry in scored):
                raise ValueError(f"Different frame span for {episode} part {part_index}")
            number = index * 10000 + part_index
            caches = [model / "quality" / f"{number}.rgb" for model in models]
            masks = region_for_part(caches, scored)
            if masks.shape != (span[1] - span[0], OUT_SIZE[1], OUT_SIZE[0]):
                raise ValueError(f"Unexpected score mask dimensions for {episode} part {part_index}")
            position = 0
            with ExitStack() as stack:
                streams = [stack.enter_context(cache.open("rb")) for cache in caches]
                for originals in reference_blocks(row, part):
                    for ref in originals:
                        mask = masks[position]
                        position += 1
                        if mask.any():
                            yield crop_group(ref, mask, streams)
                if position != len(masks) or any(stream.read(1) for stream in streams):
                    raise ValueError(f"Incomplete RGB cache alignment for {episode} part {part_index}")


def resize_groups(groups):
    # Each group contains the same crop rectangle for the reference and all methods.
    # Use pytorch-fid's bilinear resize before stacking rectangles with different sizes.
    return torch.stack([
        interpolate(torch.from_numpy(group).permute(0, 3, 1, 2).float() / 255,
                    size=(299, 299), mode="bilinear", align_corners=False)
        for group in groups
    ])


def extract_features(model, images, device):
    with torch.inference_mode():
        values = model(images.to(device))[0]
        return values.flatten(1).cpu().numpy()


def activations(model, device, groups):
    images = resize_groups(groups).flatten(0, 1)
    values = extract_features(model, images, device)
    return values.reshape(len(groups), len(REFERENCE_AND_METHODS), DIMENSIONS)


def parallel_activations(models, devices, pool, groups):
    images = resize_groups(groups)
    count = len(groups)
    features = np.empty((count, len(REFERENCE_AND_METHODS), DIMENSIONS), dtype=np.float32)
    jobs = []
    for index, (model, device) in enumerate(zip(models, devices, strict=True)):
        # Each card extracts its own method and a disjoint share of the common reference.
        start, stop = (count * shift // len(METHODS) for shift in (index, index + 1))
        inputs = torch.cat((images[:, index + 1], images[start:stop, 0]))
        jobs.append((start, stop, pool.submit(extract_features, model, inputs, device)))
    for index, (start, stop, job) in enumerate(jobs, start=1):
        values = job.result()
        features[:, index] = values[:count]
        features[start:stop, 0] = values[count:]
    return features


def evaluate(dataset, cohort, extract, batch_size, distance):
    manifest, models, metrics, expected = cohort
    features = np.empty((len(REFERENCE_AND_METHODS), expected, DIMENSIONS), dtype=np.float32)
    count = 0
    with torch.inference_mode():
        for group_batch in batched(cohort_crops(manifest, models, metrics), batch_size):
            stop = count + len(group_batch)
            features[:, count:stop] = extract(group_batch).transpose(1, 0, 2)
            count = stop
            if count % (32 * batch_size) == 0 or count == expected:
                print(f"{dataset} FID features {count}/{expected}", flush=True)
    if count != expected or count < 2 or not np.isfinite(features).all():
        raise ValueError(f"Invalid Inception features {dataset} {count}/{expected}")
    mean_ref = features[0].mean(axis=0, dtype=np.float64)
    cov_ref = np.cov(features[0], rowvar=False)
    values = {}
    for name, feat in zip(METHODS, features[1:], strict=True):
        value = distance(mean_ref, cov_ref, feat.mean(axis=0, dtype=np.float64), np.cov(feat, rowvar=False))
        if not np.isfinite(value) or value < -1e-4:
            raise ValueError(f"Invalid FID for {dataset} {name} {value}")
        values[name] = float(max(0., value))
        print(f"{dataset} {name} FID {values[name]:.6f} ({count} frames)", flush=True)
    return {"episodes": len(manifest), "scored_frames": count, "fid": values}


def score_datasets(args, extract, distance, devices):
    if not args.smoke:
        args.output.mkdir(parents=True)
    rows = ["| Dataset | Frames | ProPainter v0 ↓ | ProPainter v1 ↓ | STTN ↓ |",
            "|---|---:|---:|---:|---:|"]
    for dataset in args.dataset:
        roots = [args.cache_root / name for name in DATASETS[dataset]]
        cohort = load_cohort(roots)
        if args.smoke:
            with closing(cohort_crops(*cohort[:3])) as crops, torch.inference_mode():
                feat = extract([next(crops)])
            if feat.shape != (1, len(REFERENCE_AND_METHODS), DIMENSIONS) or not np.isfinite(feat).all():
                raise ValueError(f"Inception smoke test failed for {dataset}")
            print(f"{dataset} smoke passed, {cohort[-1]} scored frames available", flush=True)
            continue
        result = evaluate(dataset, cohort, extract, args.batch_size, distance)
        write_json(args.output / f"{dataset}.json", {
            **result,
            "metric": "FID on scored-region crops",
            "implementation": "pytorch-fid 0.3.0",
            "feature_dimensions": DIMENSIONS,
            "checkpoint_sha256": WEIGHT_SHA256,
            "devices": {name: str(device) for name, device in zip(METHODS, devices, strict=True)},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "prefill": "Original RGB outside the scored region; cached predictions inside",
            "crop_padding_pixels": CROP_PADDING,
            "minimum_crop_side_pixels": 32,
            "source_rgb_size": OUT_SIZE,
            "source_manifests": {name: {"file": str(root / "manifest.json"),
                                         "sha256": digest(root / "manifest.json")}
                                 for name, root in zip(METHODS, roots, strict=True)},
        })
        values = " | ".join(f"{result['fid'][name]:.6f}" for name in METHODS)
        rows.append(f"| {dataset} | {result['scored_frames']} | {values} |")
        (args.output / "results.md").write_text("\n".join(rows) + "\n")


def fid_backend(deps):
    sys.path.insert(0, str(deps / "python"))
    from pytorch_fid.fid_score import calculate_frechet_distance
    from pytorch_fid.inception import InceptionV3

    if version("pytorch-fid") != "0.3.0":
        raise ValueError("Expected pytorch-fid 0.3.0")
    hub = deps / "torch/hub"
    weight_hash = digest(hub / "checkpoints" / WEIGHT_FILENAME)
    if weight_hash != WEIGHT_SHA256:
        raise ValueError(f"Unexpected FID checkpoint SHA-256 {weight_hash}")
    torch.hub.set_dir(str(hub))
    return InceptionV3, calculate_frechet_distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--output", type=Path, help="New directory for FID reports")
    parser.add_argument("--cache-root", type=Path, default=OUTPUTS)
    parser.add_argument("--deps", type=Path, default=OUTPUTS / "inpaint_fid_deps")
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--device", default="cuda:0")
    devices.add_argument("--method-devices", nargs=len(METHODS), metavar="CUDA_DEVICE",
                         help="One CUDA device each for ProPainter v0, ProPainter v1 and STTN")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of frames per batch across all methods")
    parser.add_argument("--smoke", action="store_true", help="Extract features from the first scored frame of each dataset")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.smoke and args.output is None:
        parser.error("--output is required for a full run")
    selected = [torch.device(device) for device in (args.method_devices or [args.device])]
    if args.method_devices and (len(set(selected)) != len(METHODS) or any(
            d.type != "cuda" or d.index is None or d.index >= torch.cuda.device_count() for d in selected)):
        parser.error("--method-devices requires three distinct visible CUDA device indices")
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    InceptionV3, calculate_frechet_distance = fid_backend(args.deps)
    block = InceptionV3.BLOCK_INDEX_BY_DIM[DIMENSIONS]
    models = [InceptionV3([block], resize_input=False).to(device).eval() for device in selected]
    if args.method_devices:
        print("Inception-v3 ready " + ", ".join(
            f"{name} on {device}" for name, device in zip(METHODS, selected, strict=True)), flush=True)
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            extract = partial(parallel_activations, models, selected, pool)
            score_datasets(args, extract, calculate_frechet_distance, selected)
    else:
        extract = partial(activations, models[0], selected[0])
        score_datasets(args, extract, calculate_frechet_distance, selected * len(METHODS))


if __name__ == "__main__":
    main()
