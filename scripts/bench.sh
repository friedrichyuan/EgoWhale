#!/usr/bin/env bash
# Throughput benchmark on 8 GPUs with the node layout in cfgs/bench8_len1000.yaml: segment 1 card x 3 actors,
# inpaint 4 cards, depth 3 cards with one cuRobo actor riding on each, 4 MuJoCo render actors.
# Episodes are the config's list: outputs/len1000.txt, the designed mix of 1000 test episodes from 106 tasks.
# Writes to a fresh out_dir, including TensorBoard in tb/ and 0.5 s GPU utilization in gpu.csv.
# Usage: [EGOWHALE_BENCH_CONFIG=cfgs/....yaml] scripts/bench.sh <out_dir> [extra batch flags, e.g. --limit 200]
# Pass --episodes <list.txt> to run a rebased copy of the list where the dataset lives elsewhere.
set -euo pipefail
cd "$(dirname "$0")/.."
out=$(realpath -m "$1")
shift
if [ -e "$out" ]; then
    echo "$out exists; finished stages would be skipped" >&2
    exit 1
fi
mkdir -p "$out"
# Containers started with NVIDIA_DRIVER_CAPABILITIES=compute,utility have libEGL_nvidia but do not register it,
# so EGL only offers Mesa's software device.
if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] \
    && ! ls /usr/share/glvnd/egl_vendor.d/*nvidia* /etc/glvnd/egl_vendor.d/*nvidia* > /dev/null 2>&1 \
    && ldconfig -p | grep -q libEGL_nvidia.so.0; then
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$out/egl_nvidia.json"
    export __EGL_VENDOR_LIBRARY_FILENAMES=$(realpath "$out/egl_nvidia.json")
    echo "registered libEGL_nvidia for EGL"
fi
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -lms 500 > "$out/gpu.csv" &
sampler=$!
trap 'kill $sampler' EXIT
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -m egowhale.batch "${EGOWHALE_BENCH_CONFIG:-cfgs/bench8_len1000.yaml}" \
    --out "$(dirname "$out")" --name "$(basename "$out")" "$@" \
    2>&1 | tee "$out/batch.log"
