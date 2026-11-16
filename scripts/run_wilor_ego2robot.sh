#!/usr/bin/env bash
# 在这里修改 GPU、输入 MP4 和输出目录，然后运行本脚本。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
VIDEO="${VIDEO:-$ROOT/datasets/egodex/test/add_remove_lid/1.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/wilor_ego2robot_test}"
MAIN_PYTHON="${MAIN_PYTHON:-$ROOT/../.conda-egohand/bin/python}"
CAMERA_SOURCE="${CAMERA_SOURCE:-auto}" # MP4 使用 DA3；HDF5 默认使用自身内参
CONFIG="${CONFIG:-$ROOT/egowhale/wilor/cfgs/wilor_hand_pipeline.yaml}"
INTRINSICS_H5="${INTRINSICS_H5:-}"
INTRINSICS_JSON="${INTRINSICS_JSON:-}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONUNBUFFERED=1
cd "$ROOT"
args=(--video "$VIDEO" --output "$OUTPUT_DIR" --python "$MAIN_PYTHON" --config "$CONFIG" --camera-source "$CAMERA_SOURCE")
if [[ -n "$INTRINSICS_H5" ]]; then args+=(--intrinsics-h5 "$INTRINSICS_H5"); fi
if [[ -n "$INTRINSICS_JSON" ]]; then args+=(--intrinsics-json "$INTRINSICS_JSON"); fi
exec "$MAIN_PYTHON" -u tools/run_wilor_ego2robot.py "${args[@]}" "$@"
