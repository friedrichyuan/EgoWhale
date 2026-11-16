#!/usr/bin/env bash
# 修改下面的 GPU 编号、输入目录和输出目录；每张卡处理不同视频。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
INPUT_DIR="${INPUT_DIR:-$ROOT/datasets/egodex/test/add_remove_lid}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/wilor_add_remove_lid}"
MAIN_PYTHON="${MAIN_PYTHON:-$ROOT/../.conda-egohand/bin/python}"
CONFIG="${CONFIG:-$ROOT/egowhale/wilor/cfgs/wilor_hand_pipeline.yaml}"
CAMERA_SOURCE="${CAMERA_SOURCE:-da3}"   # MP4 固定使用 DA3 估计相机
MAX_VIDEOS="${MAX_VIDEOS:-0}"            # 0 = 全部；可先设 1 测试
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-10}"
cd "$ROOT"
exec "$MAIN_PYTHON" -u tools/batch_wilor_ego2robot.py --input "$INPUT_DIR" --output-root "$OUTPUT_DIR" \
  --gpu-ids "$GPU_IDS" --python "$MAIN_PYTHON" --config "$CONFIG" --camera-source "$CAMERA_SOURCE" \
  --max-videos "$MAX_VIDEOS" --heartbeat "$HEARTBEAT_SECONDS" "$@"
