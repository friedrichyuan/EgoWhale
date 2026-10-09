#!/usr/bin/env bash
# Check headless MuJoCo EGL rendering on every visible GPU, the way batch rendering selects a card.
# Each card gets its own process: the ARX scene is rendered as RGB, segmentation, and depth.
# Prints the OpenGL renderer per card and exits nonzero if any card fails.
# Usage: [CUDA_VISIBLE_DEVICES=0,1,...] scripts/check_egl.sh
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=, read -ra cards <<< "$CUDA_VISIBLE_DEVICES"
else
    mapfile -t cards < <(nvidia-smi --query-gpu=index --format=csv,noheader)
fi
failed=0
for card in "${cards[@]}"; do
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=$card python - <<'EOF'
import os
import mujoco
import numpy as np
from OpenGL import GL

model = mujoco.MjModel.from_xml_path("assets/mujoco_arx_scene/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model, height=240, width=320)
renderer.update_scene(data)
rgb = renderer.render()
gl = GL.glGetString(GL.GL_RENDERER).decode()
renderer.enable_segmentation_rendering()
renderer.update_scene(data)
segment = renderer.render()
renderer.disable_segmentation_rendering()
renderer.enable_depth_rendering()
renderer.update_scene(data)
depth = renderer.render()
renderer.close()
geoms = len(np.unique(segment[..., 0])) - 1
ok = rgb.std() > 0 and geoms > 0 and np.isfinite(depth).all() and depth.min() < depth.max()
print(f"card {os.environ['MUJOCO_EGL_DEVICE_ID']}: {'ok' if ok else 'FAIL'}  {gl}  rgb std {rgb.std():.1f}  geoms {geoms}  depth {depth.min():.2f}-{depth.max():.2f} m")
raise SystemExit(0 if ok else 1)
EOF
    status=$?
    if [ $status -ne 0 ]; then
        echo "card $card: FAIL (exit $status)"
        failed=1
    fi
done
exit $failed
