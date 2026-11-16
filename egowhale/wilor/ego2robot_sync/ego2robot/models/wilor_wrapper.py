"""WiLoR (Wild hand Localization and Reconstruction) model wrapper.

Uses WiLoR's built-in YOLO detector + ViT-based MANO regression for 3D hand
reconstruction.

Output format:
  {left/right: {frame_ids, joints_cam(T,21,3), global_orient, hand_pose, betas, transl}}
"""

import os
import sys
import logging
import numpy as np
import cv2
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

WILOR_DEFAULT_ROOT = os.environ.get(
    "WILOR_ROOT",
    str(Path(__file__).resolve().parents[2] / "third_party" / "WiLoR"),
)


class WiLoRWrapper:
    """Hand detection + 3D reconstruction using WiLoR (CVPR 2025).

    1. YOLO hand detection (per-frame, with handedness classification)
    2. ViTDetDataset preprocessing (crop, resize, normalize)
    3. WiLoR forward pass → MANO params + 3D joints + mesh vertices
    4. cam_crop_to_full → camera-frame 3D positions

    Joints output in OpenPose order (matches pipeline: thumb=1-4, index=5-8).
    """

    def __init__(self, wilor_root: str = WILOR_DEFAULT_ROOT, device: str = "cuda:0",
                 det_thresh: float = 0.3):
        self.wilor_root = wilor_root
        self.device = device
        self.det_thresh = det_thresh

        self._model = None
        self._model_cfg = None
        self._detector = None
        self._scaled_focal = None

    def _ensure_loaded(self):
        if self._model is not None:
            return

        if self.wilor_root not in sys.path:
            sys.path.insert(0, self.wilor_root)

        prev_cwd = os.getcwd()
        os.chdir(self.wilor_root)
        try:
            from wilor.models import load_wilor
            from ultralytics import YOLO

            ckpt = os.path.join(self.wilor_root, "pretrained_models", "wilor_final.ckpt")
            cfg = os.path.join(self.wilor_root, "pretrained_models", "model_config.yaml")
            det_path = os.path.join(self.wilor_root, "pretrained_models", "detector.pt")

            logger.info(f"Loading WiLoR from {ckpt}")
            self._model, self._model_cfg = load_wilor(ckpt, cfg)
            self._model = self._model.to(self.device).eval()

            logger.info(f"Loading YOLO detector from {det_path}")
            self._detector = YOLO(det_path)
        finally:
            os.chdir(prev_cwd)

    def predict(self, frames_rgb: list, focal_length: float,
                img_center: tuple = None) -> dict:
        """Run WiLoR on video frames.

        Args:
            frames_rgb: list of (H, W, 3) uint8 RGB arrays.
            focal_length: scalar focal length in pixels.
            img_center: (cx, cy) principal point. If None, uses image center.

        Returns:
            dict with 'left' and 'right' hand data.
        """
        self._ensure_loaded()

        from wilor.utils import recursive_to
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils.renderer import cam_crop_to_full

        T = len(frames_rgb)
        H, W = frames_rgb[0].shape[:2]

        if img_center is None:
            cx, cy = W / 2.0, H / 2.0
        else:
            cx, cy = img_center

        left_results = {}
        right_results = {}

        for t in range(T):
            img_rgb = frames_rgb[t]
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            # Detect hands
            with torch.no_grad(), torch.amp.autocast("cuda"):
                detections = self._detector(img_bgr, conf=self.det_thresh,
                                             verbose=False)[0]

            if len(detections.boxes) == 0:
                continue

            boxes = detections.boxes.xyxy.cpu().numpy()
            handedness = detections.boxes.cls.cpu().numpy()

            # Filter out hands that touch image borders (unreliable estimation)
            border_margin = 10
            valid_mask = (
                (boxes[:, 0] > border_margin) &
                (boxes[:, 1] > border_margin) &
                (boxes[:, 2] < W - border_margin) &
                (boxes[:, 3] < H - border_margin)
            )
            if not valid_mask.any():
                continue
            boxes = boxes[valid_mask]
            handedness = handedness[valid_mask]

            # Override handedness by bbox position (more reliable for overhead views)
            for bi in range(len(boxes)):
                box_cx = (boxes[bi][0] + boxes[bi][2]) / 2.0
                handedness[bi] = 1.0 if box_cx >= W / 2.0 else 0.0  # right=1 if on right side  # 0=left, 1=right

            # Run WiLoR
            dataset = ViTDetDataset(self._model_cfg, img_bgr, boxes, handedness,
                                     rescale_factor=2.0)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=16,
                                                       shuffle=False, num_workers=0)

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    out = self._model(batch)

                B = out['pred_keypoints_3d'].shape[0]
                for n in range(B):
                    is_right = batch['right'][n].cpu().item()
                    joints = out['pred_keypoints_3d'][n].cpu().numpy()
                    verts = out['pred_vertices'][n].cpu().numpy()

                    # Flip x for left hands (WiLoR processes all as right)
                    joints[:, 0] = (2 * is_right - 1) * joints[:, 0]
                    verts[:, 0] = (2 * is_right - 1) * verts[:, 0]

                    # Camera translation: crop frame → full image frame
                    box_center = batch['box_center'][n]
                    box_size = batch['box_size'][n]
                    img_size = batch['img_size'][n]

                    # Use WiLoR's own scaled focal (not GT focal) for crop-to-full transform
                    img_w = batch['img_size'][n][0].cpu().item()
                    img_h = batch['img_size'][n][1].cpu().item()
                    model_focal = self._model_cfg.EXTRA.FOCAL_LENGTH
                    model_img_size = self._model_cfg.MODEL.IMAGE_SIZE
                    scaled_focal = model_focal / model_img_size * max(img_w, img_h)
                    if self._scaled_focal is None:
                        self._scaled_focal = scaled_focal
                    cam_t_full = cam_crop_to_full(
                        out['pred_cam'][n:n+1], box_center[None], box_size,
                        img_size[None], scaled_focal
                    )[0].cpu().numpy()

                    # Translate joints/verts to camera frame
                    joints_cam = joints + cam_t_full
                    verts_cam = verts + cam_t_full

                    # Extract MANO params
                    mano_params = out['pred_mano_params']
                    global_orient = mano_params['global_orient'][n].cpu().numpy()  # (1,3,3)
                    hand_pose = mano_params['hand_pose'][n].cpu().numpy()  # (15,3,3)
                    betas = mano_params['betas'][n].cpu().numpy()  # (10,)

                    # Convert rotation matrices to axis-angle for compatibility
                    from scipy.spatial.transform import Rotation
                    go_aa = Rotation.from_matrix(global_orient.reshape(-1, 3, 3)).as_rotvec()
                    hp_aa = Rotation.from_matrix(hand_pose.reshape(-1, 3, 3)).as_rotvec()

                    entry = {
                        "betas": betas,
                        "global_orient": go_aa.flatten(),
                        "hand_pose": hp_aa.flatten(),
                        "transl": cam_t_full,
                        "joints_cam": joints_cam,
                        "verts_cam": verts_cam,
                    }

                    if is_right >= 0.5:
                        right_results[t] = entry
                    else:
                        left_results[t] = entry

        # Format output
        result = {}
        for hand_name, hand_results in [("left", left_results), ("right", right_results)]:
            if not hand_results:
                result[hand_name] = self._empty_hand()
                continue

            frame_ids = sorted(hand_results.keys())
            joints_cam = np.stack([hand_results[f]["joints_cam"] for f in frame_ids])
            orient_list = [hand_results[f]["global_orient"] for f in frame_ids]
            pose_list = [hand_results[f]["hand_pose"] for f in frame_ids]
            betas_list = [hand_results[f]["betas"] for f in frame_ids]
            transl_list = [hand_results[f]["transl"] for f in frame_ids]

            result[hand_name] = {
                'frame_ids': frame_ids,
                'global_orient': orient_list,
                'hand_pose': pose_list,
                'betas': betas_list,
                'transl': transl_list,
                'joints_cam': joints_cam,
                'projection_focal': self._scaled_focal,
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
        del self._model, self._detector
        self._model = self._detector = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
