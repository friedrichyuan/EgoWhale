#!/usr/bin/env bash
# Render the robot, then composite it onto the inpainted frames.
# Render needs the gripper, base, IK, and approach outputs; composite adds inpaint, masks, and depth.
# Writes visual/render.npz and visual/composite.mp4.
# Usage: scripts/composite.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.step import ROOT, run
from egowhale.visual.composite import Composite
from egowhale.visual.render import Render
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (Render, Composite))
' "$@"
