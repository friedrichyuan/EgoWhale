#!/usr/bin/env bash
# 读取仓库的 wilor.yaml，启动 WiLoR 分卡批处理。
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/egowhale/wilor/cfgs/wilor.yaml"
PYTHON="${EGOWHALE_PYTHON:-${MAIN_PYTHON:-$ROOT/../.conda-egohand/bin/python}}"

if [[ $# -eq 1 && ( "$1" == "--help" || "$1" == "-h" ) ]]; then
    echo "用法：bash scripts/wilor.sh [--dry-run]"
    echo "配置文件：$CONFIG"
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
    echo "用法：bash scripts/wilor.sh [--dry-run]。运行参数请填写 $CONFIG" >&2
    exit 2
fi

cd "$ROOT"
export PYTHONUNBUFFERED=1
exec "$PYTHON" -u tools/split_wilor_batch.py --config "$CONFIG" --python "$PYTHON" "$@"
