#!/usr/bin/env bash
# Remove the person with ProPainter.
# Needs visual/masks.npz. Writes visual/inpaint.mp4.
# Usage: scripts/inpaint.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.step import ROOT, run
from egowhale.visual.inpaint import Inpaint
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (Inpaint,))
' "$@"
