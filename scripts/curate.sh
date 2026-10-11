#!/usr/bin/env bash
# Filter the episode: frame checks, action statistics, then the VLM audit.
# Needs gripper, base, IK, approach, and visual/composite.mp4.
# Writes action/quality.json. The VLM call waits for EGOWHALE_VLM_API_KEY.
# Usage: scripts/curate.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps curate
