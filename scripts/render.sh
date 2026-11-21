#!/usr/bin/env bash
# Render only; upstream action files must already exist.
# Usage: scripts/render.sh <episode.hdf5> [out_dir] [--robot arx|piper]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale "$@" --steps render
