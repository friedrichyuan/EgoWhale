#!/usr/bin/env bash
# Run EgoWhale batch on the downloaded EgoDex Test Set.
# Usage: ./start.sh [limit] [episodes.txt|episode.hdf5|dataset_dir] [out_dir]
# A limit of 0 runs every episode. Defaults: 0 and datasets/egodex/test_episodes.txt.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /export/ra/heqijun/anaconda3/etc/profile.d/conda.sh
conda activate ego
cd "$repo"

limit="${1:-${EGOWHALE_LIMIT:-0}}"
episodes="${2:-${EGOWHALE_EPISODES:-$repo/datasets/egodex/test_episodes.txt}}"
out="${3:-/data/datasets/hqj/EgoWhale/outputs/bench_test_$(date +%Y%m%d_%H%M%S)}"

if [[ ! "$limit" =~ ^[0-9]+$ ]]; then
    echo "limit must be a nonnegative integer: $limit" >&2
    exit 2
fi
if [ ! -f "$episodes" ] && [ ! -d "$episodes" ]; then
    echo "episodes source missing: $episodes" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${CUDA_VISIBLE_DEVICES%%,*}}"
# Sample physical resources by actor class; disable Ray function profiling.
export RAY_PROFILING=0
export EGOWHALE_BENCH_CONFIG="cfgs/bench8_sttn_balanced.yaml"

bash scripts/bench.sh "$out" --episodes "$episodes" --limit "$limit"