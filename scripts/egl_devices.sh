#!/usr/bin/env bash
# List the EGL devices a process sees, under three CUDA_VISIBLE_DEVICES settings.
# MUJOCO_EGL_DEVICE_ID indexes this list. NVIDIA devices show their CUDA index; a Mesa software
# device means rendering would fall back to the CPU.
# Usage: scripts/egl_devices.sh
set -uo pipefail
cd "$(dirname "$0")/.."
list() {
    python - <<'EOF'
import ctypes
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from mujoco.egl import egl_ext as EGL

query_string = ctypes.CFUNCTYPE(ctypes.c_char_p, EGL.EGLDeviceEXT, EGL.EGLint)(EGL.eglGetProcAddress("eglQueryDeviceStringEXT"))
query_attrib = ctypes.CFUNCTYPE(EGL.EGLBoolean, EGL.EGLDeviceEXT, EGL.EGLint, ctypes.POINTER(ctypes.c_long))(
    EGL.eglGetProcAddress("eglQueryDeviceAttribEXT")
)
devices = EGL.eglQueryDevicesEXT(64)
print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')!r}: {len(devices)} EGL devices")
for index, device in enumerate(devices):
    extensions = query_string(device, EGL.EGL_EXTENSIONS)
    extensions = extensions.decode() if extensions else ""
    cuda = ""
    if "EGL_NV_device_cuda" in extensions:
        value = ctypes.c_long()
        query_attrib(device, 0x323A, ctypes.pointer(value))
        cuda = f"cuda {value.value}"
    kind = "NVIDIA" if cuda else "software (Mesa)" if "MESA_device_software" in extensions else "other"
    print(f"    EGL {index}: {kind} {cuda}  [{extensions}]")
EOF
}
echo "unset:"
(unset CUDA_VISIBLE_DEVICES; list)
echo "empty, as Ray gives actors without a GPU:"
CUDA_VISIBLE_DEVICES= list
echo "one card:"
CUDA_VISIBLE_DEVICES=1 list
