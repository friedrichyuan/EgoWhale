#!/usr/bin/env bash
# Search a shared base pose and solve the arm trajectory.
# Needs action/gripper.npz. Writes action/base.json and action/ik.npz.
# Usage: scripts/base_ik.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.action.base_ik import BaseIK
from egowhale.step import ROOT, run
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (BaseIK,))
' "$@"
