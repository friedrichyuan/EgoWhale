#!/usr/bin/env bash
# Independent streaming episode IK; inputs are existing episode output directories.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m egowhale.action.episode_ik "$@"
