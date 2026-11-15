#!/usr/bin/env bash
# 修改下面三个设置后运行：bash scripts/run_wilor_batch.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 递归查找目录下的 MP4；相对路径从 EgoWhale 仓库根目录开始。
DATASET_DIR="datasets/egodex"
OUTPUT_ROOT="outputs/wilor_batch"
GPU_ID="0"

# 1：跳过本批处理已成功且输入未变化的视频；0：重新处理所有视频。
SKIP_COMPLETED="1"
# 1：单个视频失败立即停止；0：记录失败并继续。显存不足始终停止。
STOP_ON_ERROR="0"

CONDA_ENV_PREFIX="${EGOWHALE_CONDA_PREFIX:-/home/fyc/miniconda3/envs/egowhale-wilor}"
PYTHON="${EGOWHALE_PYTHON:-$CONDA_ENV_PREFIX/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python 不存在：$PYTHON" >&2
    exit 1
fi
export PYTHONUNBUFFERED=1
exec "$PYTHON" -u tools/run_wilor_batch.py --dataset "$DATASET_DIR" --output-root "$OUTPUT_ROOT" \
    --gpu "$GPU_ID" --skip-completed "$SKIP_COMPLETED" --stop-on-error "$STOP_ON_ERROR" "$@"
