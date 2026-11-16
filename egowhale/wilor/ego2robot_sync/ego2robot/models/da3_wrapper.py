"""DA3 (Depth Anything 3) model wrapper.

Provides a unified interface for depth, camera pose, and intrinsics estimation.
Default model: DA3-NESTED (most accurate).  Configurable via ``model_name``.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Model name → directory name mapping
_MODEL_DIRS = {
    "da3giant": "DA3-GIANT",
    "da3nested": "DA3NESTED-GIANT-LARGE",
    "da3large": "DA3-LARGE-1.1",
    "da3metric": "DA3METRIC-LARGE",
}

DA3_DEFAULT_MODEL_ROOT = os.environ.get(
    "DA3_MODEL_ROOT",
    str(Path(__file__).resolve().parents[2] / "third_party" / "depth-anything-3" / "models"),
)


class DA3Wrapper:
    """Unified depth / pose / intrinsics estimator powered by DA3.

    Usage::

        wrapper = DA3Wrapper(model_name="da3nested", device="cuda:0",
                             model_root="/path/to/models")
        result = wrapper.predict(images, process_res=504)
        # result['depth']       (N, H, W)
        # result['c2w']         (N, 4, 4)  camera-to-world
        # result['intrinsics']  (N, 3, 3)
        # result['focal']       float, mean focal length in pixels
        # result['hfov']        float, horizontal FOV in radians
    """

    def __init__(
        self,
        model_name: str = "da3nested",
        device: str = "cuda:0",
        model_root: str = DA3_DEFAULT_MODEL_ROOT,
    ):
        self.model_name = model_name
        self.device = device

        # Lazy import DA3
        da3_src = os.path.normpath(os.path.join(model_root, "../src"))
        if da3_src not in sys.path:
            sys.path.insert(0, da3_src)
        os.environ.setdefault("HF_HOME", os.path.join(model_root, "../hf_cache"))

        from depth_anything_3.api import DepthAnything3

        dir_name = _MODEL_DIRS.get(model_name)
        if dir_name is None:
            raise ValueError(f"Unknown DA3 model: {model_name}. Choose from {list(_MODEL_DIRS)}")
        model_dir = os.path.join(model_root, dir_name)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"DA3 model not found: {model_dir}")

        logger.info(f"Loading DA3 [{model_name}] from {model_dir}")
        self.model = DepthAnything3.from_pretrained(model_dir).to(device=device)

    @staticmethod
    def _to_pil(images):
        from PIL import Image
        out = []
        for img in images:
            if isinstance(img, np.ndarray):
                out.append(Image.fromarray(img))
            else:
                out.append(img)
        return out

    def predict(
        self,
        images: List[np.ndarray],
        process_res: int = 504,
        ref_view_strategy: str = "middle",
    ) -> dict:
        """Run DA3 inference on a list of RGB images.

        Returns a dict with keys: depth, c2w, w2c, intrinsics, focal, hfov.
        """
        pil_images = self._to_pil(images)
        prediction = self.model.inference(
            image=pil_images,
            process_res=process_res,
            ref_view_strategy=ref_view_strategy,
        )

        result = {
            "depth": prediction.depth,  # (N, H_proc, W_proc)
            "is_metric": prediction.is_metric,
        }

        # ── Camera pose (extrinsics) ─────────────────────
        if prediction.extrinsics is not None:
            c2ws = []
            w2cs = []
            for i in range(len(prediction.extrinsics)):
                w2c = np.eye(4, dtype=np.float32)
                ext = prediction.extrinsics[i]
                w2c[:3, :] = ext[:3, :]
                w2cs.append(w2c)
                c2ws.append(np.linalg.inv(w2c))

            c2ws = np.stack(c2ws)
            w2cs = np.stack(w2cs)

            # ── Set world origin to scene center ──────────
            # DA3 pose is metric (consistent with depth) but origin = camera 0.
            # For robot_inpaint, we need a world frame where the scene (table)
            # is near the origin. We shift the origin to the scene center:
            #   scene_center = R_0 @ (0, 0, median_depth) + t_0
            # Then rebase all c2ws so scene_center becomes (0,0,0).
            depths = prediction.depth
            if depths is not None and len(depths) > 0:
                H_d, W_d = depths.shape[1], depths.shape[2]
                K0 = prediction.intrinsics[0] if prediction.intrinsics is not None else None
                if K0 is not None:
                    cx_d, cy_d = K0[0, 2], K0[1, 2]
                    fx_d, fy_d = K0[0, 0], K0[1, 1]
                    # Scene center = unproject center pixel at median depth
                    center_depth = float(np.median(depths[0, H_d//4:3*H_d//4, W_d//4:3*W_d//4]))
                    scene_pt_cam = np.array([
                        (W_d/2 - cx_d) / fx_d * center_depth,
                        (H_d/2 - cy_d) / fy_d * center_depth,
                        center_depth
                    ])
                    R0 = c2ws[0, :3, :3]
                    t0 = c2ws[0, :3, 3]
                    scene_center_world = R0 @ scene_pt_cam + t0

                    # Shift all translations so scene center = origin
                    c2ws[:, :3, 3] -= scene_center_world
                    w2cs = np.stack([np.linalg.inv(c) for c in c2ws])
                    logger.info(f"World origin set to scene center (depth={center_depth:.3f}m), "
                                f"camera norms: [{np.linalg.norm(c2ws[:,:3,3],axis=1).min():.3f}, "
                                f"{np.linalg.norm(c2ws[:,:3,3],axis=1).max():.3f}]")

            result["c2w"] = c2ws    # (N, 4, 4) metric, origin = scene center
            result["w2c"] = w2cs    # (N, 4, 4)
        else:
            result["c2w"] = None
            result["w2c"] = None

        # ── Intrinsics ───────────────────────────────────
        if prediction.intrinsics is not None and len(prediction.intrinsics) > 0:
            result["intrinsics"] = prediction.intrinsics  # (N, 3, 3) at proc resolution
            # Scale to original image resolution
            H_orig, W_orig = images[0].shape[:2]
            H_proc, W_proc = prediction.depth.shape[1], prediction.depth.shape[2]
            K0 = prediction.intrinsics[0].copy()
            K0[0, :] *= W_orig / W_proc
            K0[1, :] *= H_orig / H_proc
            fx = K0[0, 0]
            fy = K0[1, 1]
            result["intrinsics_orig"] = K0  # at original resolution
            result["focal"] = float((fx + fy) / 2)
            result["hfov"] = float(2 * np.arctan(W_orig / (2 * fx)))
            result["vfov"] = float(2 * np.arctan(H_orig / (2 * fy)))
        else:
            result["intrinsics"] = None
            result["intrinsics_orig"] = None
            result["focal"] = None
            result["hfov"] = None
            result["vfov"] = None

        # ── Confidence / sky ─────────────────────────────
        result["conf"] = prediction.conf
        result["sky"] = prediction.sky

        return result

    def release(self):
        """Free GPU memory."""
        import torch
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
