"""Seconds, fps, and peak GPU memory of every stage, run one after another on one GPU, per episode.

    python tools/stage_timing.py --episodes timing.txt --out DIR

The episodes are the HDF5 paths in the list, in order (tools/episode_list.py picks them from the designed list).
Model loading is timed apart from compute. One JSON line per stage goes to DIR/timing.jsonl, and a table to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from egowhale.__main__ import STEPS  # noqa: E402
from egowhale.pieces import missing, plan  # noqa: E402
from egowhale.step import release_gpu  # noqa: E402
from egowhale.visual.inpaint import Inpaint  # noqa: E402


def time_episode(src: Path, dst: Path, inpaint_model: str) -> list[dict]:
    import torch

    spans = plan(src, dst)
    frames = spans[-1][1]
    rows = []
    for given in STEPS:
        step = Inpaint(inpaint_model) if given is Inpaint else given()
        row = {"episode": f"{src.parent.name}/{src.stem}", "frames": frames, "pieces": len(spans), "stage": step.name}
        if not missing(dst, step.makes):
            row["status"] = "already done"
            rows.append(row)
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        try:
            started = time.perf_counter()
            step.load()
            loaded = time.perf_counter()
            step.run(src, dst)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            done = time.perf_counter()
            row.update(status="ok", load_s=round(loaded - started, 2), run_s=round(done - loaded, 2), fps=round(frames / max(done - loaded, 1e-9), 1))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:400])
        finally:
            if torch.cuda.is_available():
                row["peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            step.release()
            release_gpu()
        rows.append(row)
        if row["status"] == "failed":
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True, help="text file of HDF5 paths")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--inpaint-model", default="sttn")
    args = parser.parse_args()
    episodes = [Path(line.strip()) for line in args.episodes.read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "timing.jsonl"
    print(f"{'episode':<44}{'frames':>7}{'pieces':>7}  {'stage':<10}{'load s':>8}{'run s':>8}{'fps':>8}{'peak GB':>9}  status")
    for src in episodes:
        dst = args.out / "episodes" / f"{src.parent.name}_{src.stem}"
        for row in time_episode(src, dst, args.inpaint_model):
            with log.open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"{row['episode']:<44}{row['frames']:>7}{row['pieces']:>7}  {row['stage']:<10}"
                f"{row.get('load_s', ''):>8}{row.get('run_s', ''):>8}{row.get('fps', ''):>8}{row.get('peak_gb', ''):>9}  "
                f"{row['status']} {row.get('error', '')}",
                flush=True,
            )


if __name__ == "__main__":
    main()
