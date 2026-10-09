#!/usr/bin/env bash
# Plan a smooth move from the zero configuration to the first IK frame.
# Needs action/ik.npz. Writes action/prefix.npz.
# Usage: scripts/approach.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.action.approach import Approach
from egowhale.step import ROOT, run
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (Approach,))
' "$@"
