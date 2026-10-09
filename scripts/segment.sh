#!/usr/bin/env bash
# Segment the person with SAM 3.
# Reads the .mp4 next to the HDF5. Writes visual/masks.npz.
# Usage: scripts/segment.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.step import ROOT, run
from egowhale.visual.segment import Segment
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (Segment,))
' "$@"
