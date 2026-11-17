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

    def predict(self, frames_rgb: list, focal_length: float,
                img_center: tuple = None) -> dict:
        """Run AnyHand on video frames.

        Uses split detect → override handedness → reconstruct flow so that
        the model processes each hand with correct image flip.
        """
        report_progress("WiLoR 模型加载", message="加载 AnyHand/WiLoR 权重" if self._predictor is None else "复用 AnyHand/WiLoR 模型")
        self._ensure_loaded()

        T = len(frames_rgb)
        H, W = frames_rgb[0].shape[:2]

        left_results = {}
        right_results = {}

        for t in range(T):
            if t % 30 == 0:
                report_progress("WiLoR 逐帧预测", t, T)
            img_bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)

            # Step 1: detect hands (YOLO)
            boxes, is_right, scores = self._predictor._detect_hands(img_bgr)
            if len(boxes) == 0:
                continue

            # Step 2: override handedness by bbox center position
            for i in range(len(boxes)):
                box_cx = (boxes[i][0] + boxes[i][2]) / 2.0
                is_right[i] = box_cx >= W / 2.0

            # Step 3: reconstruct with corrected handedness
            hands = self._predictor._run_wilor(img_bgr, boxes, is_right, scores)

            for hand in hands:
                # Rescale cam_t.Z from AnyHand's internal focal (hand.focal_length,
                # typically 2500 for WiLoR backend on 640x480 input) to the real
                # camera focal so joints_cam is in true metric units. Without this
                # rescale, downstream IK targets end up ~6x too far in depth.
                z_scale = focal_length / hand.focal_length
                cam_t_scaled = hand.cam_t.copy()
                cam_t_scaled[2] *= z_scale
                joints_cam = hand.keypoints_3d + cam_t_scaled[None, :]
                verts_cam = hand.vertices + cam_t_scaled[None, :]

                entry = {
                    "betas": hand.mano_shape,
                    "global_orient": hand.mano_pose[:3],
                    "hand_pose": hand.mano_pose[3:],
                    "transl": hand.cam_t,
                    "joints_cam": joints_cam,
                    # Keep the original model camera for 2D projection. The
                    # metric-scaled depth above uses a different focal length.
                    "projection_joints_cam": hand.keypoints_3d + hand.cam_t[None, :],
                    "vertices": hand.vertices,
                    "verts_cam": verts_cam,
                    "projection_focal": hand.focal_length,
                    "score": hand.score,
                }

                target = right_results if hand.is_right else left_results
                if t not in target or hand.score > target[t]["score"]:
                    target[t] = entry

        report_progress("WiLoR 逐帧预测", T, T)
        result = {}
        for hand_name, hand_results in [("left", left_results), ("right", right_results)]:
            if not hand_results:
                result[hand_name] = self._empty_hand()
                continue

            frame_ids = sorted(hand_results.keys())
            joints_cam = np.stack([hand_results[f]["joints_cam"] for f in frame_ids])
            verts_cam = np.stack([hand_results[f]["verts_cam"] for f in frame_ids])

            result[hand_name] = {
                'frame_ids': frame_ids,
                'global_orient': [hand_results[f]["global_orient"] for f in frame_ids],
                'hand_pose': [hand_results[f]["hand_pose"] for f in frame_ids],
                'betas': [hand_results[f]["betas"] for f in frame_ids],
                'transl': [hand_results[f]["transl"] for f in frame_ids],
                'joints_cam': joints_cam,
                'projection_joints_cam': np.stack([hand_results[f]["projection_joints_cam"] for f in frame_ids]),
                'verts_cam': verts_cam,
                'projection_focal': hand_results[frame_ids[0]]["projection_focal"],
            }

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
