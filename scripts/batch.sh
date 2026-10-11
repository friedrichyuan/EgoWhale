#!/usr/bin/env bash
# Run many episodes on Ray, with the node layout of a config in cfgs/. Unset keys take egowhale/config.py DEFAULTS.
# Composite and curation run on CPU. The VLM audit is not called.
# GPU pool sizes must add up to the devices you expose. Finished stages are skipped.
# Usage: scripts/batch.sh <config.yaml> [--name NAME] [--limit N] [--episodes PATH...] [--out DIR]
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale.batch "$@"
