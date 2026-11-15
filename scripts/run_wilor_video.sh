#!/usr/bin/env bash
# 修改下方三个设置后运行：bash scripts/run_wilor_video.sh
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 在这里修改输入视频、输出根目录和 GPU 编号。
# 相对路径以 EgoWhale 仓库根目录为起点。
VIDEO="datasets/egodex/test/add_remove_lid/0.mp4"
OUTPUT_ROOT="outputs/wilor_video"
GPU_ID="0"
# Default: the entire WiLoR node, including SAM3/tracking/DynHaMR/HMP/gaps.
# raw is an explicit debugging mode only.
MODE="${MODE:-full}"
# Empty: automatically use the sibling .hdf5 ONLY for the GT video, after inference.
GT_HDF5="${GT_HDF5:-}"
CONDA_ENV_PREFIX="${EGOWHALE_CONDA_PREFIX:-/home/fyc/miniconda3/envs/egowhale-wilor}"
if [[ -x "$CONDA_ENV_PREFIX/bin/python" ]]; then
    MODEL_PYTHON="${MODEL_PYTHON:-$CONDA_ENV_PREFIX/bin/python}"
else
    MODEL_PYTHON="${MODEL_PYTHON:-}"
fi
DA3_PYTHON="${DA3_PYTHON:-${MODEL_PYTHON:-$ROOT/thirdparty/.venvs/wilor-da3/bin/python}}"
WILOR_PYTHON="${WILOR_PYTHON:-${MODEL_PYTHON:-$ROOT/thirdparty/WiLoR/.venv/bin/python}}"
SAM3_PYTHON="${SAM3_PYTHON:-${MODEL_PYTHON:-$ROOT/thirdparty/.venvs/wilor-vision/bin/python}}"
DYNHAMR_PYTHON="${DYNHAMR_PYTHON:-${MODEL_PYTHON:-$ROOT/thirdparty/.venvs/wilor-dynhamr/bin/python}}"
DA3_SAMPLES="${DA3_SAMPLES:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PYTHON="${EGOWHALE_PYTHON:-${MODEL_PYTHON:-$ROOT/.venv/bin/python}}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Main Python not found: $PYTHON" >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$PYTHON" -u tools/run_wilor_video.py \
    --video "$VIDEO" --output-root "$OUTPUT_ROOT" --gpu "$GPU_ID" --mode "$MODE" \
    --gt "$GT_HDF5" --da3-python "$DA3_PYTHON" --wilor-python "$WILOR_PYTHON" \
    --sam3-python "$SAM3_PYTHON" --dynhamr-python "$DYNHAMR_PYTHON" \
    --da3-samples "$DA3_SAMPLES" --batch-size "$BATCH_SIZE" "$@"
