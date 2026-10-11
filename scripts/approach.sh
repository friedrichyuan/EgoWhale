#!/usr/bin/env bash
# Plan a smooth move from the zero configuration to the first IK frame.
# Needs action/ik.npz. Writes action/prefix.npz.
# Upstream stages are not rerun; a missing input raises FileNotFoundError.
# Usage: scripts/approach.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps approach
