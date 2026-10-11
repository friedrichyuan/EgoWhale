#!/usr/bin/env bash
# Render the robot, then composite it onto the inpainted frames.
# Render needs the gripper, base, IK, and approach outputs; composite adds inpaint, masks, and depth.
# Writes visual/render.npz and visual/composite.mp4.
# Usage: scripts/composite.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps render composite
