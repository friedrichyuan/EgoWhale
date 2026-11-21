#!/usr/bin/env bash
# Batch-test one node with persistent workers on the selected devices.
# Usage: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 bash scripts/node_bench.sh base_ik --episodes list.txt --inputs-root DIR --out NEW_DIR [--robot piper] [--workers N]
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then
  if [[ ${1:-} == composite ]]; then
    export CUDA_VISIBLE_DEVICES=-1
  else
    CUDA_VISIBLE_DEVICES="$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd, -)"
    export CUDA_VISIBLE_DEVICES
  fi
fi

exec python -m egowhale.node_bench "$@"
