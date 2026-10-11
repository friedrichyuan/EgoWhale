#!/usr/bin/env bash
# Remove the person with STTN. Add --inpaint-model propainter for ProPainter.
# Needs visual/masks.npz. Writes visual/inpaint.mp4.
# Upstream stages are not rerun; a missing input raises FileNotFoundError.
# Usage: scripts/inpaint.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps inpaint
