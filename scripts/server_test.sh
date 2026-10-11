#!/usr/bin/env bash
# Server acceptance test for EgoWhale on an 8-GPU node. Every episode comes from the designed test list,
# outputs/len1000.txt (1000 episodes, 106 tasks), in its own order. Runs, in order, and keeps going past a failure:
#   0 environment, files, and the episode list    1 EGL on every card    2 unit tests
#   3 per-stage timing on one GPU (3 episodes from the list)
#   4 batch smoke run (24 episodes from the list, short to long, 8 GPUs)
#   5 throughput benchmark per layout on the whole list (or its first BENCH_LIMIT), GPU utilization every 0.5 s
# Everything a reviewer needs goes to <out_dir>/ and to <out_dir>.tgz: logs, summaries, gpu.csv, batch.jsonl,
# TensorBoard, timing, and three smoke composites. Episode outputs stay in <out_dir>/ and are not packed.
#
# Usage: [EGODEX=<egodex_test_dir>] [EPISODES=outputs/len1000.txt] [BENCH_LIMIT=N] \
#        [LAYOUTS="bench8_len1000 bench8_sttn"] [SKIP_TIMING=1] [SKIP_BENCH=1] scripts/server_test.sh <out_dir>
# EGODEX rebases the list's paths onto <dir>/<task>/<episode>.hdf5; leave it unset when the list's paths are real.
# Activate the Python environment first; `python` on PATH is what runs.
set -uo pipefail
cd "$(dirname "$0")/.."
repo=$(pwd)
if [ $# -lt 1 ]; then
    sed -n '2,15p' "$0"
    exit 2
fi
out=$(realpath -m "$1")
episodes=$(realpath -m "${EPISODES:-outputs/len1000.txt}")
if [ -e "$out" ]; then
    echo "$out exists; pick a fresh out_dir" >&2
    exit 1
fi
mkdir -p "$out"
bench_limit=${BENCH_LIMIT:-}
layouts=${LAYOUTS:-bench8_len1000 bench8_sttn}
summary="$out/summary.txt"
echo "EgoWhale server test  $(date -Is)  episodes $episodes  egodex ${EGODEX:-as listed}  bench_limit ${bench_limit:-all}  layouts $layouts" \
    | tee "$summary"

# stage NAME LOG CMD...: run CMD with output to LOG, and note its exit code and minutes in the summary.
stage() {
    local name=$1 log=$2
    shift 2
    echo "== $name  (log $log)"
    local started=$SECONDS
    "$@" > "$out/$log" 2>&1
    local code=$?
    printf "%-28s exit %-3s %6s min  %s\n" "$name" "$code" "$(awk "BEGIN {printf \"%.1f\", ($SECONDS - $started) / 60}")" "$log" \
        | tee -a "$summary"
    return $code
}

# Tracebacks from Ray worker logs since the run started, so actor failures are not lost.
ray_errors() {
    local logs=${RAY_TMPDIR:-/tmp/ray}/session_latest/logs
    [ -d "$logs" ] || { echo "no Ray logs at $logs"; return; }
    grep -l "Traceback" "$logs"/*.err 2>/dev/null | head -20 | while read -r file; do
        echo "---- $file"
        grep -n -A25 "Traceback" "$file" | tail -60
    done
}

# Containers started with NVIDIA_DRIVER_CAPABILITIES=compute,utility have libEGL_nvidia but do not register it.
if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] \
    && ! ls /usr/share/glvnd/egl_vendor.d/*nvidia* /etc/glvnd/egl_vendor.d/*nvidia* > /dev/null 2>&1 \
    && ldconfig -p | grep -q libEGL_nvidia.so.0; then
    echo '{"file_format_version":"1.0.0","ICD":{"library_path":"libEGL_nvidia.so.0"}}' > "$out/egl_nvidia.json"
    export __EGL_VENDOR_LIBRARY_FILENAMES="$out/egl_nvidia.json"
    echo "registered libEGL_nvidia for EGL" | tee -a "$summary"
fi

environment() {
    echo "## host"; hostname; uname -a; date -Is
    echo "## git"; git -C "$repo" log -1 --format="%h %s" 2>/dev/null; git -C "$repo" status --short 2>/dev/null | head -40
    echo "## cpu / memory / disk"; nproc; free -g; df -h "$repo" "$out" ${EGODEX:+"$EGODEX"} 2>/dev/null
    echo "## gpus"; nvidia-smi; nvidia-smi topo -m
    echo "## env"; env | grep -E "^(CUDA|NVIDIA|MUJOCO|__EGL|PYTORCH|RAY|EGOWHALE)" | sed 's/\(API_KEY=\).*/\1<set>/'
    echo "## python"; which python; python - <<'EOF'
import importlib, sys
print(sys.version)
for name in ("torch", "torchvision", "numpy", "scipy", "cv2", "h5py", "PIL", "mujoco", "ray", "curobo", "omegaconf",
             "yaml", "loguru", "tensorboard", "safetensors", "timm", "pytest"):
    try:
        module = importlib.import_module(name)
        print(f"{name:<12} {getattr(module, '__version__', '?')}")
    except Exception as exc:
        print(f"{name:<12} MISSING ({type(exc).__name__}: {exc})")
import torch
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"  cuda:{index} {props.name} {props.total_memory / 2**30:.1f} GiB  sm {props.major}.{props.minor}")
EOF
    echo "## files"
    local missing=0
    for path in \
        thirdparty/sam3/weights/sam3/sam3.pt thirdparty/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz \
        thirdparty/cutie/tracker thirdparty/cutie/weights/cutie-base-mega.pth \
        thirdparty/sttn/model/sttn.py thirdparty/sttn/core/spectral_norm.py thirdparty/sttn/checkpoints/sttn.pth \
        thirdparty/da3/src thirdparty/da3/weights/DA3-GIANT/model.safetensors \
        assets/curobo_arx/arx_acone.yml assets/curobo_arx/arx_acone_kin.urdf assets/mujoco_arx_scene/scene.xml; do
        if [ -e "$path" ]; then echo "ok       $path"; else echo "MISSING  $path"; missing=1; fi
    done
    for path in thirdparty/propainter/weights/ProPainter.pth thirdparty/propainter/weights/raft-things.pth; do
        if [ -e "$path" ]; then echo "ok       $path (optional)"; else echo "absent   $path (optional, ProPainter only)"; fi
    done
    return $missing
}
stage "0 environment" 00_env.log environment

# The designed list, rebased and checked, plus the smoke and timing subsets drawn from it. Nothing runs without it.
if ! stage "0 episode list" 00_episodes.log python tools/episode_list.py "$episodes" "$out/episodes" ${EGODEX:+--data "$EGODEX"}; then
    tee -a "$summary" < "$out/00_episodes.log"
    echo "stopping: the episode list is missing or incomplete" | tee -a "$summary"
    exit 1
fi
tee -a "$summary" < "$out/00_episodes.log"

stage "1 egl per card" 01_egl.log scripts/check_egl.sh
stage "2 unit tests" 02_pytest.log python -m pytest -q tests

if [ -z "${SKIP_TIMING:-}" ]; then
    stage "3 stage timing (GPU 0)" 03_timing.log env CUDA_VISIBLE_DEVICES=0 \
        python tools/stage_timing.py --episodes "$out/episodes/timing.txt" --out "$out/timing"
fi

# Fit each layout to this host's RAM and CPUs. The GPU layout is left as it is.
mkdir -p "$out/cfgs"
for layout in $layouts; do
    stage "  fit $layout" "fit_$layout.log" python tools/fit_config.py "cfgs/$layout.yaml" "$out/cfgs/$layout.yaml"
done

first=${layouts%% *}
if [ -f "$out/cfgs/$first.yaml" ]; then
    started=$SECONDS
    stage "4 batch smoke ($first)" 04_smoke.log \
        python -m egowhale.batch "$out/cfgs/$first.yaml" --episodes "$out/episodes/smoke.txt" --out "$out/batch" --name smoke
    smoke_wall=$((SECONDS - started))
    ray_errors > "$out/04_ray_errors.log" 2>&1
    python tools/batch_summary.py "$out/batch/smoke" --wall "$smoke_wall" > "$out/04_smoke_summary.txt" 2>&1
    cat "$out/04_smoke_summary.txt" | head -20 | tee -a "$summary"
else
    echo "4 batch smoke skipped: $first does not fit this host (see fit_$first.log)" | tee -a "$summary"
fi

if [ -z "${SKIP_BENCH:-}" ]; then
    for layout in $layouts; do
        if [ ! -f "$out/cfgs/$layout.yaml" ]; then
            echo "5 bench $layout skipped: does not fit this host" | tee -a "$summary"
            continue
        fi
        ( while sleep 600; do echo "#### $(date -Is)"; scripts/diagnose.sh; done ) > "$out/05_diagnose_$layout.log" 2>&1 &
        watcher=$!
        started=$SECONDS
        stage "5 bench $layout" "05_bench_$layout.log" env EGOWHALE_BENCH_CONFIG="$out/cfgs/$layout.yaml" \
            scripts/bench.sh "$out/bench_$layout" --episodes "$out/episodes/episodes.txt" ${bench_limit:+--limit "$bench_limit"}
        wall=$((SECONDS - started))
        kill $watcher 2>/dev/null
        ray_errors > "$out/05_ray_errors_$layout.log" 2>&1
        python tools/batch_summary.py "$out/bench_$layout" --gpu "$out/bench_$layout/gpu.csv" --wall "$wall" \
            > "$out/05_summary_$layout.txt" 2>&1
        tee -a "$summary" < "$out/05_summary_$layout.txt"
    done
fi

# Pack what review needs: no episode outputs except three smoke composites.
mkdir -p "$out/samples"
# <run>/<task>/<episode>/visual/composite.mp4 -> samples/<task>_<episode>.mp4
find "$out/batch/smoke" -name composite.mp4 2>/dev/null | head -3 | while read -r video; do
    episode=$(dirname "$(dirname "$video")")
    cp "$video" "$out/samples/$(basename "$(dirname "$episode")")_$(basename "$episode").mp4"
done
(
    cd "$(dirname "$out")"
    name=$(basename "$out")
    find "$name" -maxdepth 1 -type f -print
    find "$name/timing" -name timing.jsonl -print 2>/dev/null
    find "$name/cfgs" "$name/samples" "$name/episodes" -type f -print 2>/dev/null
    for run in "$name/batch/smoke" "$name"/bench_*; do
        [ -d "$run" ] || continue
        find "$run" -maxdepth 1 -type f \( -name "*.jsonl" -o -name "*.yaml" -o -name "*.csv" -o -name "*.log" \) -print
        find "$run/tb" -type f -print 2>/dev/null
    done
) > "$out/pack.list"
tar -czf "$out.tgz" -C "$(dirname "$out")" -T "$out/pack.list"
echo "packed $(wc -l < "$out/pack.list") files into $out.tgz ($(du -h "$out.tgz" | cut -f1))" | tee -a "$summary"
