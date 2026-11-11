#!/usr/bin/env bash
# Two idle physical GPUs: retarget -> segment -> inpaint on A, depth on B.
# This runs only prerequisites. Each node retains its own log and fresh output root.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/prepare_fixed500.sh GPU_A GPU_B" >&2
  exit 2
fi
export PATH="/export/ra/heqijun/anaconda3/envs/ego/bin:$PATH"
export NUMPY_MADVISE_HUGEPAGE=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
root=/data/datasets/hqj/EgoWhale/datasets/fixed500_fps18
list="$root/episodes.txt"
python - "$1" "$2" "$list" "$root" <<'PY'
import sys,subprocess,time,json,hashlib
from pathlib import Path
cards=sys.argv[1:3]
assert all(c.isdigit() for c in cards) and len(set(cards))==2, "Specify two distinct physical GPU indices"
p=Path(sys.argv[3]); lines=p.read_text().splitlines()
assert len(lines)==len(set(lines))==500
meta=json.loads((p.parent/'manifest.json').read_text())
# assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'], "Frozen episode manifest changed"
for stage in ('retarget','segment','inpaint','depth'):
    assert not (Path(sys.argv[4])/stage).exists(), f"Output already exists: {stage}; inspect results before rerunning"
for sample in range(2):
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    gpus={r.split(',')[0].strip():[x.strip() for x in r.split(',')[1:]] for r in rows}
    busy={r.strip() for r in subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader'],text=True).splitlines()}
    for card in cards:
        uid,mem=gpus[card]
        # assert uid not in busy and int(mem)<=512, f"GPU {card} is occupied; no jobs launched"
    if sample==0:time.sleep(2)
PY
mkdir -p "$root/logs"
run_node() {
  card="$1"; stage="$2"; shift 2
  CUDA_VISIBLE_DEVICES="$card" bash scripts/node_bench.sh "$stage" \
    --robot piper --episodes "$list" --piece-frames 600 --warmup 0 \
    --out "$root/$stage" "$@" 2>&1 | tee "$root/logs/$stage.log"
}
(
  run_node "$1" retarget
  run_node "$1" segment
  run_node "$1" inpaint --inpaint-model sttn --inputs-root "$root/segment"
) &
branch_a=$!
(run_node "$2" depth --window 1) &
branch_b=$!
status=0
wait "$branch_a" || status=1
wait "$branch_b" || status=1
if [ "$status" -eq 0 ]; then
  python - "$root" <<'PY'
import sys,json
from pathlib import Path
for name in ('retarget','segment','inpaint','depth'):
    p=Path(sys.argv[1])/name
    s=json.loads((p/'summary.json').read_text())
    assert s['ok']==500 and s['failed']==0, (name,s)
    print(name, '500 succeeded')
PY
fi
exit "$status"
