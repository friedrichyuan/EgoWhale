"""Episode lists for the server test, all drawn from one designed list (default outputs/len1000.txt).

    python tools/episode_list.py LIST OUT_DIR [--data EGODEX_TEST_DIR]

The list's paths are rebased onto --data as <data>/<task>/<episode>.hdf5 when given, and every episode is
checked for its HDF5 and .mp4. Writes, under OUT_DIR:
  episodes.txt  the whole list in its own order, rebased    smoke.txt   24 episodes for the batch smoke run
  timing.txt    3 episodes for per-stage timing             lengths.tsv frames per episode
Smoke and timing episodes stay within SHORT_LIMIT frames so those steps finish in minutes; both include
episodes longer than one piece. Exits 1 if any listed episode is missing.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

SMOKE = 24
# Smoke and timing skip the few very long episodes; the benchmark still runs them.
SHORT_LIMIT = 1200
PIECE = 600


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("list", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--data", type=Path, help="EgoDex test directory to rebase the list's paths onto")
    args = parser.parse_args()
    if args.data:
        args.data = args.data.expanduser().resolve()
    if not args.list.is_file():
        sys.exit(f"{args.list} not found; copy the designed episode list there, or set EPISODES")
    listed = [Path(line.strip()) for line in args.list.read_text().splitlines() if line.strip()]
    paths = [args.data / path.parent.name / path.name if args.data else path for path in listed]
    missing = [path for path in paths if not (path.is_file() and path.with_suffix(".mp4").is_file())]
    print(f"{args.list}: {len(paths)} episodes, {len(set(paths))} distinct, {len({p.parent.name for p in paths})} tasks")
    if args.data:
        print(f"rebased from {Counter(str(p.parents[1]) for p in listed).most_common(1)[0][0]} onto {args.data}")
    if missing:
        print(f"{len(missing)} missing, e.g. {[str(p) for p in missing[:5]]}")
        sys.exit(1)
    lengths = []
    for path in paths:
        with h5py.File(path) as handle:
            lengths.append(int(handle["transforms/camera"].shape[0]))
    frames = np.array(lengths)
    print(
        f"frames total {frames.sum()}  min {frames.min()}  median {int(np.median(frames))}  "
        f"p90 {int(np.quantile(frames, 0.9))}  max {frames.max()}  over {PIECE}: {(frames > PIECE).sum()}  "
        f"over {SHORT_LIMIT}: {(frames > SHORT_LIMIT).sum()}"
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "episodes.txt").write_text("".join(f"{path}\n" for path in paths))
    (args.out / "lengths.tsv").write_text("".join(f"{path}\t{count}\n" for path, count in zip(paths, lengths)))

    short = sorted((count, index) for index, count in enumerate(lengths) if count <= SHORT_LIMIT)
    picks = [short[round(position * (len(short) - 1) / (SMOKE - 1))][1] for position in range(SMOKE)]
    # At least two episodes that are split into pieces.
    split = [index for count, index in short if count > PIECE]
    for index in split[: 2 - sum(lengths[index] > PIECE for index in picks)]:
        picks[len(picks) // 2] = index
    picks = list(dict.fromkeys(picks))
    (args.out / "smoke.txt").write_text("".join(f"{paths[index]}\n" for index in picks))
    median = short[len(short) // 2][1]
    p90 = short[int(len(short) * 0.9)][1]
    longest = short[-1][1]
    timing = list(dict.fromkeys([median, p90, longest]))
    (args.out / "timing.txt").write_text("".join(f"{paths[index]}\n" for index in timing))
    print(f"smoke {len(picks)} episodes, {sum(lengths[i] for i in picks)} frames, {sum(lengths[i] > PIECE for i in picks)} split")
    print(f"timing {[f'{paths[i].parent.name}/{paths[i].stem} ({lengths[i]})' for i in timing]}")


if __name__ == "__main__":
    main()
