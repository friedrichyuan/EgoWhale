"""python -m egowhale <episode.hdf5> [out_dir]"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

from egowhale.action.approach import Approach
from egowhale.action.base_ik import BaseIK
from egowhale.action.curate import Curate
from egowhale.action.retarget import Retarget
from egowhale.step import ROOT, run
from egowhale.visual.composite import Composite
from egowhale.visual.depth import Depth
from egowhale.visual.inpaint import Inpaint
from egowhale.visual.render import Render
from egowhale.visual.segment import Segment

STEPS = (Retarget, Segment, Inpaint, Depth, BaseIK, Approach, Render, Composite, Curate)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m egowhale <episode.hdf5> [out_dir]")
    src = Path(sys.argv[1]).expanduser().resolve()
    dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
    run(src, dst, STEPS)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
