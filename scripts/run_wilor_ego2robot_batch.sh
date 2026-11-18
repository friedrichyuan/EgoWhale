#!/usr/bin/env bash
# 兼容旧入口；GPU 分组及输入输出请在 egowhale/wilor/cfgs/wilor.yaml 修改。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for name in GPU_IDS INPUT_DIR OUTPUT_DIR DATASET_DIR OUTPUT_ROOT MAX_VIDEOS HEARTBEAT_SECONDS CONFIG; do
  if [[ -n "${!name:-}" ]]; then
    echo "旧环境变量 $name 不再生效，请在 egowhale/wilor/cfgs/wilor.yaml 的 split 中配置；视频数也可使用 --max-videos。" >&2
    exit 2
  fi
done
echo "旧批处理入口已切换为 DA3 / WiLoR 分卡调度。" >&2
exec bash "$ROOT/scripts/run_wilor_split_multigpu.sh" "$@"
