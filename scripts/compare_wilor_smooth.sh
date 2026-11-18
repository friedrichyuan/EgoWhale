#!/usr/bin/env bash
# 修改这些变量后运行：bash scripts/compare_wilor_smooth.sh
# 同一份 WiLoR 检测/跟踪结果，对比 root/smooth；MP4 固定走 DA3 / Path B。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-auto}" # auto 选空闲卡；也可改为 0、1 等编号
VIDEO="${VIDEO:-$ROOT/datasets/egodex/test/paint_clean_brush/14.mp4}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/wilor_smooth_compare/$(date +%Y%m%d_%H%M%S)_$$}"
MAIN_PYTHON="${MAIN_PYTHON:-$ROOT/../.conda-egohand/bin/python}"
CONFIG="${CONFIG:-$ROOT/egowhale/wilor/cfgs/wilor.yaml}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/egowhale-matplotlib}"
if [[ ! -x "$MAIN_PYTHON" ]]; then echo "Python 不存在：$MAIN_PYTHON" >&2; exit 1; fi
export PATH="$(dirname "$MAIN_PYTHON"):$PATH"
if [[ ! -f "$VIDEO" ]]; then echo "输入不存在：$VIDEO" >&2; exit 1; fi
if [[ "$GPU_ID" == auto ]]; then
    if ! GPU_ROWS="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)"; then
        echo "无法访问 NVIDIA 驱动；尚未运行模型。请在可访问 GPU 的服务器执行。" >&2
        exit 1
    fi
    GPU_ID="$("$MAIN_PYTHON" -c 'import sys; rows=[tuple(map(int,r.split(","))) for r in sys.stdin if r.strip()]; idle=[r for r in rows if r[1]<1024 and r[2]<10]; print(min(idle,key=lambda r:(r[1],r[2],r[0]))[0] if idle else "")' <<< "$GPU_ROWS")"
    if [[ -z "$GPU_ID" ]]; then echo "没有找到显存占用低于 1GB、利用率低于 10% 的 GPU；请手动设置 GPU_ID。" >&2; exit 1; fi
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$ROOT"
"$MAIN_PYTHON" -c 'import torch; assert torch.cuda.is_available(), "当前会话无法使用 CUDA，尚未开始实验"; print("实验 GPU:",torch.cuda.get_device_name(0))'
if [[ -e "$OUTPUT_DIR" ]]; then echo "请使用新的 OUTPUT_DIR：$OUTPUT_DIR" >&2; exit 1; fi
mkdir -p "$OUTPUT_DIR"
printf 'GPU=%s | 视频=%s | 输出=%s\n' "$GPU_ID" "$VIDEO" "$OUTPUT_DIR"
"$MAIN_PYTHON" -u egowhale/wilor/ego2robot_sync/scripts/run_hand_experiment.py \
    --video "$VIDEO" --output "$OUTPUT_DIR/work" --config "$CONFIG" \
    --path b --camera-source da3 --video-only --enable-dynhamr --stage full \
    2>&1 | tee "$OUTPUT_DIR/process.log"
"$MAIN_PYTHON" -u tools/compare_wilor_smooth.py \
    --video "$VIDEO" --work "$OUTPUT_DIR/work" --output "$OUTPUT_DIR" \
    2>&1 | tee -a "$OUTPUT_DIR/process.log"
