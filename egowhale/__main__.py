"""python -m egowhale <episode.hdf5> [out_dir] [--steps NAME ...] [--inpaint-model sttn|propainter]"""

from __future__ import annotations

import argparse
from pathlib import Path

from egowhale.action.approach import Approach
from egowhale.action.base_ik import BaseIK
from egowhale.action.retarget import Retarget
from egowhale.paths import default_out
from egowhale.quality.curate import Curate
from egowhale.robots.factory import make_step
from egowhale.step import run
from egowhale.visual.composite import Composite
from egowhale.visual.depth import Depth
from egowhale.visual.inpaint import DEFAULT_MODEL, MODELS, Inpaint
from egowhale.visual.render import Render
from egowhale.visual.segment import Segment

STEPS = (Retarget, Segment, Inpaint, Depth, BaseIK, Approach, Render, Composite, Curate)


def main() -> None:
    names = [step.name for step in STEPS]
    parser = argparse.ArgumentParser(prog="python -m egowhale", description="Run the pipeline on one episode in this process.")
    parser.add_argument("episode", type=Path, help="EgoDex HDF5 file with a sibling .mp4")
    parser.add_argument("out", type=Path, nargs="?", help="output directory; default outputs/<task>/<episode>")
    parser.add_argument("--steps", nargs="+", choices=names, metavar="NAME", help=f"run only these, in pipeline order: {' '.join(names)}")
    parser.add_argument("--inpaint-model", choices=MODELS, default=DEFAULT_MODEL)
    parser.add_argument("--robot", choices=("arx", "piper", "dual_piper"), default="arx")
    parser.add_argument("--no-approach", action="store_true", help="do not generate an approach prefix")
    args = parser.parse_args()
    src = args.episode.expanduser().resolve()
    dst = args.out.expanduser().resolve() if args.out else default_out(src)
    chosen = [step for step in STEPS if args.steps is None or step.name in args.steps]
    if args.no_approach:
        chosen = [step for step in chosen if step is not Approach]
    run(src, dst, tuple(Inpaint(args.inpaint_model) if step is Inpaint else make_step(step, args.robot) for step in chosen))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
