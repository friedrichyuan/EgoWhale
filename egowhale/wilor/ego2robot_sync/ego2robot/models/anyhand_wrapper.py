"""AnyHand model wrapper — fine-tuned HaMeR/WiLoR for 3D hand reconstruction.

Uses AnyHandPredictor (unified API) with checkpoints fine-tuned on 2.5M
synthetic hand images for better generalization.

Splits detect → override handedness → reconstruct so that ViTDetDataset flips
images correctly for each hand.
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
from pathlib import Path
from ego2robot.utils.progress import report_progress

logger = logging.getLogger(__name__)

ANYHAND_DEFAULT_ROOT = os.environ.get(
    "ANYHAND_ROOT",
    str(Path(__file__).resolve().parents[2] / "third_party" / "AnyHand"),
)


class AnyHandWrapper:
    """Hand detection + 3D reconstruction using AnyHand fine-tuned models."""

    def __init__(self, anyhand_root: str = ANYHAND_DEFAULT_ROOT, device: str = "cuda:0",
                 det_thresh: float = 0.3, backend: str = "wilor"):
        self.anyhand_root = anyhand_root
        self.device = device
        self.det_thresh = det_thresh
        self.backend = backend
        self._predictor = None

    def _ensure_loaded(self):
        if self._predictor is not None:
            return

        if self.anyhand_root not in sys.path:
            sys.path.insert(0, self.anyhand_root)

        prev_cwd = os.getcwd()
        os.chdir(self.anyhand_root)
        try:
            from ego2robot.utils.legacy_mano import prepare_mano_loading
            prepare_mano_loading()
            # Upstream also distributes the predictor as scripts/rgb_predictor.py.
            predictor_file = Path(self.anyhand_root) / "scripts" / "rgb_predictor.py"
            if predictor_file.is_file():
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "ego2robot_anyhand_predictor", predictor_file)
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                AnyHandPredictor = module.AnyHandPredictor
            else:
                from anyhand import AnyHandPredictor
            logger.info(f"Loading AnyHand (backend={self.backend}) from {self.anyhand_root}")
            self._predictor = AnyHandPredictor(
                backend=self.backend,
                device=self.device,
                det_conf=self.det_thresh,
            )
        finally:
            os.chdir(prev_cwd)

    def candidates(self, frames_rgb, count):
        """Yield all hands before camera calibration or temporal filtering."""
        report_progress("WiLoR 模型加载", message="加载 AnyHand/WiLoR 权重" if self._predictor is None else "复用 AnyHand/WiLoR 模型")
        self._ensure_loaded()
        run = self._predictor._run_wilor if self.backend == "wilor" else self._predictor._run_hamer
        for t, frame in enumerate(frames_rgb):
            if t % 30 == 0:
                report_progress("WiLoR 逐帧预测", t, count)
            img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            boxes, is_right, scores = self._predictor._detect_hands(img_bgr)
            for i, box in enumerate(boxes):
                is_right[i] = (box[0] + box[2]) / 2.0 >= frame.shape[1] / 2.0
            yield run(img_bgr, boxes, is_right, scores) if len(boxes) else []
        report_progress("WiLoR 逐帧预测", count, count)

    def predict(self, frames_rgb: list, focal_length: float, img_center: tuple = None):
        return self.calibrate(self.candidates(frames_rgb, len(frames_rgb)), focal_length, include_mesh=True)

    @classmethod
    def calibrate(cls, candidates, focal_length, *, include_mesh=False):
        """Choose one candidate per side and apply the basic branch's focal correction."""
        left_results = {}
        right_results = {}
        for t, hands in enumerate(candidates):
            for hand in hands:
                # Rescale cam_t.Z from AnyHand's internal focal (hand.focal_length,
                # typically 2500 for WiLoR backend on 640x480 input) to the real
                # camera focal so joints_cam is in true metric units. Without this
                # rescale, downstream IK targets end up ~6x too far in depth.
                z_scale = focal_length / hand.focal_length
                cam_t_scaled = hand.cam_t.copy()
                cam_t_scaled[2] *= z_scale
                joints_cam = hand.keypoints_3d + cam_t_scaled[None, :]

                entry = {
                    "betas": hand.mano_shape,
                    "global_orient": hand.mano_pose[:3],
                    "hand_pose": hand.mano_pose[3:],
                    "transl": hand.cam_t,
                    "joints_cam": joints_cam,
                    # Keep the original model camera for 2D projection. The
                    # metric-scaled depth above uses a different focal length.
                    "projection_joints_cam": hand.keypoints_3d + hand.cam_t[None, :],
                    "projection_focal": hand.focal_length,
                    "score": hand.score,
                }

                if include_mesh:
                    entry["verts_cam"] = hand.vertices + cam_t_scaled[None, :]

                target = right_results if hand.is_right else left_results
                if t not in target or hand.score > target[t]["score"]:
                    target[t] = entry

        result = {}
        for hand_name, hand_results in [("left", left_results), ("right", right_results)]:
            if not hand_results:
                result[hand_name] = cls._empty_hand()
                continue

            frame_ids = sorted(hand_results.keys())
            joints_cam = np.stack([hand_results[f]["joints_cam"] for f in frame_ids])

            result[hand_name] = {
                'frame_ids': frame_ids,
                'global_orient': [hand_results[f]["global_orient"] for f in frame_ids],
                'hand_pose': [hand_results[f]["hand_pose"] for f in frame_ids],
                'betas': [hand_results[f]["betas"] for f in frame_ids],
                'transl': [hand_results[f]["transl"] for f in frame_ids],
                'joints_cam': joints_cam,
                'projection_joints_cam': np.stack([hand_results[f]["projection_joints_cam"] for f in frame_ids]),
                'projection_focal': hand_results[frame_ids[0]]["projection_focal"],
            }

            if include_mesh:
                result[hand_name]["verts_cam"] = np.stack([hand_results[f]["verts_cam"] for f in frame_ids])

        return result

    @staticmethod
    def _empty_hand():
        return {
            'frame_ids': [],
            'global_orient': [],
            'hand_pose': [],
            'betas': [],
            'transl': [],
            'joints_cam': None,
        }

    def release(self):
        del self._predictor
        self._predictor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
