#!/usr/bin/env bash
# Estimate metric scene depth with Depth Anything 3.
# Needs visual/inpaint.mp4. Writes visual/depth.npz.
# Usage: scripts/depth.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.step import ROOT, run
from egowhale.visual.depth import Depth
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (Depth,))
' "$@"
