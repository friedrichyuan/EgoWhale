"""Summarise one batch run for review: python tools/batch_summary.py <run dir> [--gpu gpu.csv] [--wall SECONDS]

Reads batch.jsonl (one line per episode), the TensorBoard scalars in tb/, and optionally the nvidia-smi samples
that scripts/bench.sh writes. Prints plain text.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def episodes(run: Path) -> None:
    rows = [json.loads(line) for line in (run / "batch.jsonl").read_text().splitlines() if line.strip()]
    ok = [row for row in rows if row["ok"]]
    print(f"episodes {len(rows)}  ok {len(ok)}  failed {len(rows) - len(ok)}")
    frames = sum(row["frames"] for row in rows)
    pieces = Counter(row["pieces"] for row in rows)
    print(f"frames {frames}  pieces per episode {dict(sorted(pieces.items()))}")
    if ok:
        seconds = np.array([row["seconds"] for row in ok])
        print(f"episode wall seconds  median {np.median(seconds):.1f}  p90 {np.quantile(seconds, 0.9):.1f}  max {seconds.max():.1f}")
    errors = Counter()
    example = {}
    for row in rows:
        if row["ok"]:
            continue
        stage, _, message = row["error"].partition(": ")
        key = (stage.split("[")[0], message[:120])
        errors[key] += 1
        example.setdefault(key, row["episode"])
    for (stage, message), count in errors.most_common(15):
        print(f"  {count:>4} x {stage:<10} {message}   e.g. {example[(stage, message)]}")
    compares = [row for row in rows if "compare" in row]
    if compares:
        print(f"compare video failed on {len(compares)}: {compares[0]['compare']}")


def board(run: Path) -> None:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("tensorboard not importable; skipped scalars")
        return
    accumulator = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0})
    accumulator.Reload()
    tags = sorted(accumulator.Tags().get("scalars", []))
    print("tensorboard, last value (max):")
    for tag in tags:
        values = [event.value for event in accumulator.Scalars(tag)]
        if values:
            print(f"  {tag:<28} {values[-1]:10.2f}  ({max(values):.2f})")


def gpus(path: Path) -> None:
    samples = defaultdict(list)
    memory = defaultdict(list)
    with path.open() as handle:
        for row in csv.reader(handle):
            if len(row) < 4:
                continue
            try:
                index = int(row[1])
                samples[index].append(float(row[2].strip().split()[0]))
                memory[index].append(float(row[3].strip().split()[0]))
            except ValueError:
                continue
    print("gpu  util mean  util p50  share<30%  mem mean GB  mem max GB")
    for index in sorted(samples):
        util = np.array(samples[index])
        mem = np.array(memory[index]) / 1024
        print(f"{index:>3}  {util.mean():9.1f}  {np.median(util):8.1f}  {(util < 30).mean():9.0%}  {mem.mean():11.1f}  {mem.max():10.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--gpu", type=Path)
    parser.add_argument("--wall", type=float, help="seconds the whole run took, for end-to-end fps")
    args = parser.parse_args()
    print(f"== {args.run}")
    if (args.run / "batch.jsonl").is_file():
        episodes(args.run)
        if args.wall:
            frames = sum(json.loads(line)["frames"] for line in (args.run / "batch.jsonl").read_text().splitlines() if line.strip())
            print(f"end-to-end {frames / args.wall:.1f} fps over {args.wall:.0f} s")
    else:
        print("no batch.jsonl")
    if (args.run / "tb").is_dir():
        board(args.run)
    if args.gpu and args.gpu.is_file():
        gpus(args.gpu)


if __name__ == "__main__":
    main()
