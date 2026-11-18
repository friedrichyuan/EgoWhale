#!/usr/bin/env bash
# 独立运行手部节点：仅 MP4 -> DA3 + Path B；有标注的 HDF5 直接输入 Retarget。
# 用法：bash scripts/wilor.sh <input.mp4> [out_dir]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}" # 修改 GPU 编号，也可通过环境变量设置
PYTHON="${EGOWHALE_PYTHON:-$ROOT/../.conda-egohand/bin/python}"
if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "用法：bash scripts/wilor.sh <input.mp4> [out_dir]" >&2
    exit 2
fi
INPUT="$1"
TASK="$(basename "$(dirname "$INPUT")")"
NAME="$(basename "${INPUT%.*}")"
OUTPUT="${2:-$ROOT/outputs/$TASK/$NAME}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1
cd "$ROOT"
exec "$PYTHON" -m egowhale.wilor --video "$INPUT" --output "$OUTPUT" --python "$PYTHON"
