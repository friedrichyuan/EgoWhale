#!/usr/bin/env bash
# Run many episodes on Ray. Defaults: segment 1, inpaint 4, depth 2, action 1.
# Composite and curation run on CPU. The VLM audit is not called.
# GPU pool sizes must add up to the devices you expose. Finished stages are skipped.
# Usage: scripts/batch.sh <episode.hdf5 | dataset-dir> [more paths...]
# Optional flags are passed through, for example --inflight 8 --limit 40
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale.batch "$@"
