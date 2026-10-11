#!/usr/bin/env bash
# Segment the person with SAM 3.
# Reads the .mp4 next to the HDF5. Writes visual/masks.npz.
# Upstream stages are not rerun; a missing input raises FileNotFoundError.
# Usage: scripts/segment.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps segment
