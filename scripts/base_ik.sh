#!/usr/bin/env bash
# Search a shared base pose and solve the arm trajectory.
# Needs action/gripper.npz. Writes action/base.json and action/ik.npz.
# Upstream stages are not rerun; a missing input raises FileNotFoundError.
# Usage: scripts/base_ik.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps base_ik
