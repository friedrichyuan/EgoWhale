#!/usr/bin/env bash
# Throughput benchmark on 8 GPUs: segment 1 card x 3 actors, inpaint 4 cards with one action actor
# riding on two of them, depth 3 cards, 4 MuJoCo render actors. Episodes are the first 1000 of 11 EgoDex tasks.
# Writes to a fresh out_dir, including TensorBoard in tb/ and 0.5 s GPU utilization in gpu.csv.
# Usage: scripts/bench.sh <egodex_test_dir> <out_dir> [extra batch flags, e.g. --limit 200]
set -euo pipefail
cd "$(dirname "$0")/.."
data=$(realpath "$1")
out=$2
shift 2
if [ -e "$out" ]; then
    echo "$out exists; finished stages would be skipped" >&2
    exit 1
fi
tasks=(
    basic_pick_place vertical_pick_place assemble_disassemble_furniture_bench_chair stack_unstack_plates
    insert_remove_furniture_bench_cabinet stack basic_fold scoop_dump_ice charge_uncharge_device
    use_rubiks_cube pick_place_food
)
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
python -m egowhale.batch "${tasks[@]/#/$data/}" --out "$out" --limit 1000 \
    --segment 1 --segment-actors 3 --inpaint 4 --action 0 --action-actors 2 --depth 3 --render 4 "$@" \
    2>&1 | tee "$out/batch.log"
