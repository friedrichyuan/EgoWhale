#!/usr/bin/env bash
# Batch-test one node on the explicitly selected device; no Ray pools.
# Usage: CUDA_VISIBLE_DEVICES=9 bash scripts/node_bench.sh base_ik --episodes list.txt --inputs-root DIR --out NEW_DIR [--robot piper]
set -euo pipefail
cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=8

python -m egowhale.node_bench "$@"
