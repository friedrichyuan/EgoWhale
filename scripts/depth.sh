#!/usr/bin/env bash
# Estimate metric scene depth with Depth Anything 3.
# Reads the .mp4 next to the HDF5. Writes visual/depth.npz.
# Upstream stages are not rerun; a missing input raises FileNotFoundError.
# Usage: scripts/depth.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps depth
