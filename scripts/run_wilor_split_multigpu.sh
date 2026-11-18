#!/usr/bin/env bash
# 独立 WiLoR 多卡入口。默认处理 egodex_fixed500_18fps 下的所有 MP4。
# dynhamr_enabled 选择 split.profiles.basic 或 split.profiles.refined。
# --dry-run 仅检查配置，--max-videos N 限制条数，--resume-run DIR 续跑。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MAIN_PYTHON="${MAIN_PYTHON:-$ROOT/../.conda-egohand/bin/python}"
SPLIT_CONFIG="${SPLIT_CONFIG:-$ROOT/egowhale/wilor/cfgs/wilor.yaml}"
export PYTHONUNBUFFERED=1
exec "$MAIN_PYTHON" -u tools/split_wilor_batch.py --config "$SPLIT_CONFIG" --python "$MAIN_PYTHON" "$@"
