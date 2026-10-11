#!/usr/bin/env bash
# Retarget hand keypoints to a smoothed camera-frame gripper.
# Writes action/gripper.npz. No upstream stage required.
# Upstream stages are not rerun; a missing input raises FileNotFoundError.
# Usage: scripts/retarget.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps retarget
