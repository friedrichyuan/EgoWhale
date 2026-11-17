#!/usr/bin/env bash
# 修改 GPU_ID、VIDEO、OUTPUT_ROOT 后运行：bash scripts/test_wilor_da3.sh
# 此脚本固定 Path B，强制由 DA3 估计内参；即使同名 HDF5 有内参也不使用。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
VIDEO="${VIDEO:-$ROOT/datasets/egodex/test/paint_clean_brush/14.mp4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs/wilor_da3_test}"
MAIN_PYTHON="${MAIN_PYTHON:-$ROOT/../.conda-egohand/bin/python}"
CONFIG="${CONFIG:-$ROOT/egowhale/wilor/cfgs/wilor.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)_$$}"
if [[ ! -x "$MAIN_PYTHON" ]]; then echo "Python 不存在：$MAIN_PYTHON" >&2; exit 1; fi
if [[ ! -f "$VIDEO" ]]; then echo "输入视频不存在：$VIDEO" >&2; exit 1; fi
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1
cd "$ROOT"
printf 'GPU=%s | 输入=%s | 相机来源=DA3 | 输出=%s\n' "$GPU_ID" "$VIDEO" "$OUTPUT_DIR"
exec "$MAIN_PYTHON" -u tools/run_wilor_ego2robot.py \
  --video "$VIDEO" --output "$OUTPUT_DIR" --python "$MAIN_PYTHON" --config "$CONFIG" \
  --camera-source da3 "$@"
