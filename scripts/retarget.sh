#!/usr/bin/env bash
# Retarget hand keypoints to a smoothed camera-frame gripper.
# Writes action/gripper.npz. No upstream stage required.
# Usage: scripts/retarget.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.action.retarget import Retarget
from egowhale.step import ROOT, run
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (Retarget,))
' "$@"
