#!/usr/bin/env bash
# Filter the episode: frame checks, action statistics, then the VLM audit.
# Needs gripper, base, IK, approach, and visual/composite.mp4.
# Writes action/quality.json. The VLM call waits for EGOWHALE_VLM_API_KEY.
# Usage: scripts/curate.sh <episode.hdf5> [out_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
python -c '
import sys
from pathlib import Path
from egowhale.action.curate import Curate
from egowhale.step import ROOT, run
src = Path(sys.argv[1]).expanduser().resolve()
dst = Path(sys.argv[2]).expanduser().resolve() if len(sys.argv) > 2 else ROOT / "outputs" / src.parent.name / src.stem
run(src, dst, (Curate,))
' "$@"
