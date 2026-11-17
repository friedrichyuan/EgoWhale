#!/usr/bin/env bash
# 先在 egowhale/wilor/cfgs/wilor.yaml 中修改分卡、数据集和视频数。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MAIN_PYTHON="${MAIN_PYTHON:-$ROOT/../.conda-egohand/bin/python}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$ROOT/egowhale/wilor/cfgs/wilor.yaml}"
export PYTHONUNBUFFERED=1
exec "$MAIN_PYTHON" -u tools/split_wilor_batch.py --config "$SPLIT_CONFIG" --python "$MAIN_PYTHON" "$@"
