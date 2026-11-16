#!/usr/bin/env bash
# Usage: prepare_fixed500.sh RETARGET_GPUS SEGMENT_GPUS [DEPTH_GPUS]
# Each argument is a comma-separated list of physical GPU indices.
# With two arguments, retarget/segment/inpaint share the first group, depth uses the second.
# This runs only prerequisites. Each node retains its own log and fresh output root.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# != 2 && $# != 3 ]]; then
  echo "Usage: bash scripts/prepare_fixed500.sh GPU_A GPU_B or RETARGET_GPUS SEGMENT_GPUS DEPTH_GPUS" >&2
  exit 2
fi
export PATH="/export/pgs/heqijun/miniconda3/envs/egowhale/bin:$PATH"
export NUMPY_MADVISE_HUGEPAGE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
retarget="$1"
segment="$1"
depth="$2"
if [[ $# == 3 ]]; then
  segment="$2"
  depth="$3"
fi
list="${EGOWHALE_FIXED500_LIST:-$PWD/datasets/egodex_fixed500_18fps/episodes.txt}"
root="${EGOWHALE_FIXED500_OUT:-$PWD/outputs/fixed500_fps18}"
python - "$list" "$root" "$@" <<'PY'
import re, subprocess, sys, time
from pathlib import Path

listing, root = map(Path, sys.argv[1:3])
paths = [Path(line) for line in listing.read_text().splitlines() if line.strip()]
if len(paths) != 500 or len(set(paths)) != 500:
    raise SystemExit("Expected 500 distinct episodes")
missing = [p for p in paths if not p.is_file() or not p.with_suffix(".mp4").is_file()]
if missing:
    raise SystemExit(f"{len(missing)} missing HDF5/video pairs, e.g. {missing[0]}")
for stage in ("retarget", "segment", "inpaint", "depth"):
    if (root / stage).exists():
        raise SystemExit(f"Output already exists: {root / stage}")
groups = sys.argv[3:]
if not all(re.fullmatch(r"[0-9]+(?:,[0-9]+)*", group) for group in groups):
    raise SystemExit("Specify comma-separated physical GPU indices")
cards = [str(int(card)) for group in groups for card in group.split(",")]
if len(cards) != len(set(cards)):
    raise SystemExit("Specify distinct physical GPUs")
for sample in range(2):
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True
    ).splitlines()
    gpu = {index: (uuid, mem) for index, uuid, mem in (map(str.strip, row.split(",")) for row in rows)}
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    for card in cards:
        if card not in gpu or gpu[card][0] in {row.strip() for row in busy} or int(gpu[card][1]) > 512:
            raise SystemExit(f"GPU {card} is not idle")
    if sample == 0:
        time.sleep(2)
PY
mkdir -p "$root/logs"
run_node() {
  local cards="$1" stage="$2"
  shift 2
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$cards" bash scripts/node_bench.sh "$stage" \
    --robot piper --episodes "$list" --piece-frames 600 --warmup 0 \
    --out "$root/$stage" "$@" 2>&1 | tee "$root/logs/$stage.log"
}
pids=()
if [[ $# == 3 ]]; then
  (run_node "$retarget" retarget) &
  pids+=("$!")
fi
(
  if [[ $# == 2 ]]; then run_node "$retarget" retarget; fi
  run_node "$segment" segment
  run_node "$segment" inpaint --inpaint-model sttn --inputs-root "$root/segment"
) &
pids+=("$!")
(run_node "$depth" depth --window 1) &
pids+=("$!")
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" == 0 ]]; then
  python - "$root" <<'PY'
import json, sys
from pathlib import Path
for stage in ("retarget", "segment", "inpaint", "depth"):
    summary = json.loads((Path(sys.argv[1]) / stage / "summary.json").read_text())
    if summary["ok"] != 500 or summary["failed"]:
        raise SystemExit(f"{stage}: incomplete results: {summary}")
    print(stage, "500 succeeded")
PY
fi
exit "$status"
