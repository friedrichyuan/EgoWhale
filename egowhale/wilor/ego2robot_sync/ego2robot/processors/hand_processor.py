"""Hand Processor — 3D hand reconstruction from raw video.

Uses focal length from cam_intrinsics.json (either GT from base stage or
predicted by DA3) to drive hand reconstruction.

Input:  cam_intrinsics.json (for focal length; from base or depth_pose stage)
        video frames (from video file)
Output: hand_processor/hand_data_left.npz, hand_data_right.npz
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path

from omegaconf import DictConfig

from ego2robot.processors.base_processor import BaseProcessor, register_processor
from ego2robot.utils.progress import report_progress, run_streamed_command

logger = logging.getLogger(__name__)


@register_processor("hand")
class HandProcessor(BaseProcessor):
    """Hand reconstruction processor.

    Reads focal length from cam_intrinsics.json (GT or DA3-predicted), then runs
    hand detection + 3D reconstruction + MANO forward kinematics.
    """

    gpu_fraction: float = 0.3

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._wrapper = None

    def _ensure_model(self):
        if self._wrapper is not None:
            return
        backend = self.cfg.get("hand_backend", "anyhand")
        det_thresh = self.cfg.get("hand_det_thresh", 0.3)
        if backend == "wilor":
            from ego2robot.models.wilor_wrapper import WiLoRWrapper
            self._wrapper = WiLoRWrapper(
                wilor_root=self.cfg.get("wilor_root",
                    os.environ.get("WILOR_ROOT", "third_party/WiLoR")),
                device="cuda:0",
                det_thresh=det_thresh,
            )
        elif backend == "anyhand":
            from ego2robot.models.anyhand_wrapper import AnyHandWrapper
            self._wrapper = AnyHandWrapper(
                anyhand_root=self.cfg.get("anyhand_root",
                    os.environ.get("ANYHAND_ROOT", "third_party/AnyHand")),
                device="cuda:0",
                det_thresh=det_thresh,
                backend=self.cfg.get("anyhand_backend", "wilor"),
            )
        else:
            raise ValueError(f"unknown hand_backend: {backend}")

    def process_one_demo(self, data_sub_folder: str) -> None:
        save_dir = Path(data_sub_folder) / "hand_processor"
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── Load focal length (GT or DA3-predicted) ───────
        fx, fy, cx, cy = self._load_intrinsics(data_sub_folder)
        if fx is None:
            raise FileNotFoundError("No focal length found; run depth_pose or provide intrinsics first")
        focal = (fx + fy) / 2.0

        # ── Load video frames ────────────────────────────
        frames_rgb, fps = self._load_frames(data_sub_folder)
        if not frames_rgb:
            raise ValueError(f"No video frames found in {data_sub_folder}")

        T = len(frames_rgb)
        H, W = frames_rgb[0].shape[:2]
        # Fall back to image centre if principal point is missing
        if cx is None: cx = W / 2.0
        if cy is None: cy = H / 2.0
        logger.info(f"[hand] {T} frames, {W}x{H}, fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

        # ── Run hand estimation ────────────────────────────
        self._ensure_model()

        # When DynHaMR is enabled with AnyHand, use full pipeline (SAM3+edge+tracking+jump+DynHaMR)
        if self.cfg.get("dynhamr_enabled", False) and self.cfg.get("hand_backend") == "anyhand":
            self._anyhand_dynhamr_full_pipeline(
                frames_rgb, save_dir, data_sub_folder, fx, fy, cx, cy, T, H, W, fps
            )
            logger.info(f"[hand] saved to {save_dir}")
            return

        result = self._wrapper.predict(frames_rgb, focal_length=focal,
                                       img_center=(cx, cy))

        # ── Save hand data ────────────────────────────────
        min_detection_rate = self.cfg.get("min_hand_detection_rate", 0.3)

        for hand_name in ["left", "right"]:
            hand = result[hand_name]
            n_det = len(hand['frame_ids'])
            detection_rate = n_det / T if T > 0 else 0

            # Filter: if detection rate too low, mark as undetected
            # (prevents noisy interpolation from sparse detections)
            if detection_rate < min_detection_rate:
                logger.info(f"[hand] {hand_name}: {n_det}/{T} = {detection_rate:.0%} < {min_detection_rate:.0%}, marking as undetected")
                n_det = 0

            # Build hand_data_*.npz arrays
            if n_det > 0 and hand['joints_cam'] is not None:
                # kpts_3d: (T_total, 21, 3) with zeros for undetected frames
                kpts_3d = np.zeros((T, 21, 3), dtype=np.float32)
                kpts_2d = np.zeros((T, 21, 2), dtype=np.float32)
                hand_detected = np.zeros(T, dtype=bool)
                for i, fid in enumerate(hand['frame_ids']):
                    if fid < T:
                        kpts_3d[fid] = hand['joints_cam'][i]
                        hand_detected[fid] = True
                        # Project 3D → 2D using appropriate focal and pp
                        proj_focal = hand.get('projection_focal')
                        if proj_focal:
                            proj_fx = proj_focal
                            proj_fy = proj_focal
                            proj_cx = W / 2.0
                            proj_cy = H / 2.0
                        else:
                            proj_fx = fx
                            proj_fy = fy
                            proj_cx = cx
                            proj_cy = cy
                        # The AnyHand projection focal belongs to its original
                        # camera translation, not the metric-scaled 3D output.
                        projected_joints = hand.get('projection_joints_cam')
                        pts = projected_joints[i] if projected_joints is not None else hand['joints_cam'][i]
                        if pts[:, 2].min() > 0:
                            kpts_2d[fid, :, 0] = pts[:, 0] / pts[:, 2] * proj_fx + proj_cx
                            kpts_2d[fid, :, 1] = pts[:, 1] / pts[:, 2] * proj_fy + proj_cy

                # EEF rotation from wrist orientation
                eef_rot = np.zeros((T, 3, 3), dtype=np.float32)
                for i, fid in enumerate(hand['frame_ids']):
                    if fid < T and i < len(hand['global_orient']):
                        from scipy.spatial.transform import Rotation
                        R = Rotation.from_rotvec(hand['global_orient'][i]).as_matrix()
                        eef_rot[fid] = R

                # Robust jump filtering: iterative + short-segment removal
                hand_detected, n_filtered = self._filter_hand_jumps(
                    hand_detected, kpts_3d, kpts_2d, eef_rot, hand_name, T, fps=fps, focal=focal
                )

                # Re-check detection rate after filtering
                remaining_rate = hand_detected.sum() / T
                if remaining_rate < min_detection_rate:
                    logger.info(f"[hand] {hand_name}: after filtering {hand_detected.sum()}/{T} = "
                                f"{remaining_rate:.0%} < {min_detection_rate:.0%}, marking all as undetected")
                    hand_detected[:] = False
                    kpts_3d[:] = 0
                    kpts_2d[:] = 0
                    eef_rot[:] = 0
            else:
                kpts_3d = np.zeros((T, 21, 3), dtype=np.float32)
                kpts_2d = np.zeros((T, 21, 2), dtype=np.float32)
                hand_detected = np.zeros(T, dtype=bool)
                eef_rot = np.zeros((T, 3, 3), dtype=np.float32)

            np.savez_compressed(
                str(save_dir / f"hand_data_{hand_name}.npz"),
                hand_detected=hand_detected,
                kpts_2d=kpts_2d,
                kpts_3d=kpts_3d,
                eef_rot=eef_rot,
                frame_indices=np.arange(T),
                # Also save raw MANO params for downstream use
                mano_orient=[np.array(o) for o in hand['global_orient']] if hand['global_orient'] else [],
                mano_pose=[np.array(p) for p in hand['hand_pose']] if hand['hand_pose'] else [],
                mano_betas=[np.array(b) for b in hand['betas']] if hand['betas'] else [],
                mano_transl=[np.array(t) for t in hand['transl']] if hand['transl'] else [],
                detected_frame_ids=hand['frame_ids'],
            )

        # ── Generate hand skeleton overlay video ─────────
        self._save_hand_overlay_video(frames_rgb, save_dir, focal, H, W, fps=fps)

        logger.info(f"[hand] saved to {save_dir}")

    def _filter_hand_jumps(self, hand_detected, kpts_3d, kpts_2d, eef_rot, hand_name, T, fps=30.0, focal=500.0):
        """Iterative jump filtering + short segment removal.

        1. Iteratively remove frames with wrist velocity > threshold
        2. Remove isolated short segments (< min_segment_len continuous frames)
        3. If remaining detection rate < threshold, mark all as undetected

        Returns: (hand_detected, n_total_filtered)
        """
        min_segment_len = self.cfg.get("min_hand_segment_len", 5)
        n_before = hand_detected.sum()

        # Step 1: Iterative velocity-based filtering (up to 3 rounds)
        # Adaptive pos_floor: reconstructed 3D positions scale inversely with focal length.
        # Reference focal ~500 (typical for 240p). Smaller focal → larger 3D coords → higher velocity.
        ref_focal = 500.0
        focal_scale = ref_focal / max(focal, 50.0)
        pos_floor = 0.9 / fps * focal_scale
        for round_i in range(3):
            detected_idx = np.where(hand_detected)[0]
            if len(detected_idx) < 3:
                break
            wrists = kpts_3d[detected_idx, 0]
            velocities = np.linalg.norm(np.diff(wrists, axis=0), axis=1)
            if len(velocities) == 0:
                break
            median_v = np.median(velocities[velocities > 0]) if np.any(velocities > 0) else 0.0
            jump_thresh = max(median_v * 4, pos_floor)

            jumped = velocities > jump_thresh
            if not jumped.any():
                break
            for j in np.where(jumped)[0]:
                fid = detected_idx[j + 1]
                hand_detected[fid] = False
                kpts_3d[fid] = 0
                kpts_2d[fid] = 0
                eef_rot[fid] = 0

        # Step 2: Remove short isolated segments
        detected_idx = np.where(hand_detected)[0]
        if len(detected_idx) > 0:
            # Find continuous segments
            segments = []
            seg_start = detected_idx[0]
            for i in range(1, len(detected_idx)):
                if detected_idx[i] != detected_idx[i-1] + 1:
                    segments.append((seg_start, detected_idx[i-1]))
                    seg_start = detected_idx[i]
            segments.append((seg_start, detected_idx[-1]))

            # Remove segments shorter than min_segment_len
            for seg_s, seg_e in segments:
                seg_len = seg_e - seg_s + 1
                if seg_len < min_segment_len:
                    for fid in range(seg_s, seg_e + 1):
                        hand_detected[fid] = False
                        kpts_3d[fid] = 0
                        kpts_2d[fid] = 0
                        eef_rot[fid] = 0

        n_filtered = int(n_before - hand_detected.sum())
        if n_filtered > 0:
            logger.info(f"[hand] {hand_name}: filtered {n_filtered} frames "
                        f"(jumps + short segments), remaining {hand_detected.sum()}/{T}")
        return hand_detected, n_filtered

    def _save_hand_overlay_video(self, frames_rgb, save_dir, focal, H, W, suffix="", data_prefix="hand_data", fps=30.0):
        """Draw detected 3D hand joints projected onto video frames.

        Output: hand_processor/video_hand_overlay.mp4 (H.264, viewable in VSCode)
        - Green skeleton = right hand detected
        - Magenta skeleton = left hand detected
        - Gray text = undetected frame
        """
        import cv2
        import subprocess

        # Load saved hand data
        hands = {}
        for side in ['left', 'right']:
            p = save_dir / f"{data_prefix}_{side}.npz"
            if p.exists():
                d = np.load(str(p), allow_pickle=True)
                hands[side] = {
                    'detected': d['hand_detected'],
                    'kpts_2d': d['kpts_2d'],
                }

        if not hands:
            return

        T = len(frames_rgb)
        status = getattr(self, "optimization_status", {})
        overlay_stage = ("优化前标注视频" if suffix else
                         "smooth 优化后标注视频" if status.get("result_source") == "smooth_fit" else
                         "回退结果标注视频" if status.get("fallback") else
                         "优化后标注视频" if status.get("succeeded") else "手部标注视频")
        report_progress(overlay_stage, 0, T)
        points_only = self.cfg.get("hand_overlay_points_only", False)
        # MANO skeleton connectivity (21 joints)
        BONES = [
            (0,1),(1,2),(2,3),(3,4),      # index
            (0,5),(5,6),(6,7),(7,8),      # middle
            (0,9),(9,10),(10,11),(11,12),  # ring
            (0,13),(13,14),(14,15),(15,16),# pinky
            (0,17),(17,18),(18,19),(19,20),# thumb
        ]
        COLORS = {'right': (0, 220, 0), 'left': (220, 0, 220)}  # green / magenta

        overlay_frames = []
        for t in range(T):
            frame = frames_rgb[t].copy()

            for side in ['left', 'right']:
                if side not in hands:
                    continue
                detected = hands[side]['detected'][t]
                kpts = hands[side]['kpts_2d'][t]  # (21, 2)

                if detected and np.any(kpts > 0):
                    color = COLORS[side]
                    # Draw bones
                    for i, j in ([] if points_only else BONES):
                        p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
                        p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
                        if 0 < p1[0] < W and 0 < p1[1] < H and 0 < p2[0] < W and 0 < p2[1] < H:
                            cv2.line(frame, p1, p2, color, 1, cv2.LINE_AA)
                    # Draw joints
                    for k in range(21):
                        px = (int(kpts[k, 0]), int(kpts[k, 1]))
                        if 0 < px[0] < W and 0 < px[1] < H:
                            cv2.circle(frame, px, 2, color, -1)
                    # Wrist label
                    wx, wy = int(kpts[0, 0]), int(kpts[0, 1])
                    if not points_only and 0 < wx < W and 0 < wy < H:
                        cv2.putText(frame, side[0].upper(), (wx + 5, wy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

            # Detection status text
            status_parts = []
            for side in ['left', 'right']:
                if side in hands:
                    det = hands[side]['detected'][t]
                    tag = f"{side[0].upper()}:{'OK' if det else '--'}"
                    status_parts.append(tag)
            status = "  ".join(status_parts)
            if not points_only:
                cv2.putText(frame, status, (5, H - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"f{t}", (W - 30, H - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1, cv2.LINE_AA)

            overlay_frames.append(frame)
            if (t + 1) % 30 == 0 or t + 1 == T:
                report_progress(overlay_stage, t + 1, T)

        # Save as H.264 MP4
        out_path = str(save_dir / f"video_hand_overlay{suffix}.mp4")
        h, w = overlay_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path + ".tmp.mp4", fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {out_path}")
        for f in overlay_frames:
            writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        writer.release()

        # Transcode to H.264 for VSCode compatibility
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", out_path + ".tmp.mp4",
                 "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                 "-pix_fmt", "yuv420p", out_path],
                capture_output=True, timeout=300, check=True
            )
            os.remove(out_path + ".tmp.mp4")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            os.rename(out_path + ".tmp.mp4", out_path)

        logger.info(f"[hand] overlay video: {out_path}")

    def _load_intrinsics(self, data_sub_folder: str):
        """Load (fx, fy, cx, cy) from cam_intrinsics.json saved at demo root.

        Returns (None, None, None, None) if no intrinsics file is found.
        Falls back to (focal, focal, None, None) for legacy files that only
        stored a scalar focal length.
        """
        intr_path = os.path.join(data_sub_folder, "cam_intrinsics.json")
        if not os.path.exists(intr_path):
            # Fallback for legacy layout where files were under depth_pose_processor/
            intr_path = os.path.join(data_sub_folder, "depth_pose_processor", "cam_intrinsics.json")
        if not os.path.exists(intr_path):
            return None, None, None, None

        with open(intr_path) as f:
            data = json.load(f)

        src = data.get("left") or data   # older files nest under "left"
        fx = src.get("fx")
        fy = src.get("fy", fx)
        cx = src.get("cx")
        cy = src.get("cy")
        # Legacy: only a scalar 'focal' available
        if fx is None and data.get("focal"):
            fx = fy = float(data["focal"])
        return (float(fx) if fx is not None else None,
                float(fy) if fy is not None else None,
                float(cx) if cx is not None else None,
                float(cy) if cy is not None else None)

    def _load_frames(self, data_sub_folder: str):
        """Load video frames (same logic as depth_pose_processor)."""
        import av
        for name in ["video_L.mp4", "video_rgb_imgs.mkv"]:
            vpath = os.path.join(data_sub_folder, name)
            if os.path.exists(vpath):
                break
        else:
            # Find any video
            for f in sorted(os.listdir(data_sub_folder)):
                if f.endswith((".mp4", ".mkv")) and not f.startswith("video_annot"):
                    vpath = os.path.join(data_sub_folder, f)
                    break
            else:
                return None, 30

        container = av.open(vpath)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        container.close()
        return frames, fps

    def _anyhand_dynhamr_full_pipeline(self, frames_rgb, save_dir, data_sub_folder,
                                        fx, fy, cx, cy, T, H, W, fps):
        """Full pipeline: AnyHand + SAM3 + edge + tracking + jump + DynHaMR."""
        import cv2
        import shutil

        hand_mode = self.cfg.get("hand_mode", "full")

        # Check for raw checkpoint — skip detection if available
        raw_left = save_dir / "hand_data_raw_left.npz"
        raw_right = save_dir / "hand_data_raw_right.npz"
        if raw_left.exists() and raw_right.exists():
            report_progress("恢复已有 WiLoR 预测", message="复用检测结果，继续完整优化")
            if hand_mode == "detect_only":
                logger.info("[hand_detect] Raw checkpoint already exists, nothing to do")
                return
            logger.info("[hand] Raw checkpoint found, skipping AnyHand detection")
            for side in ["left", "right"]:
                shutil.copy2(str(save_dir / f"hand_data_raw_{side}.npz"),
                             str(save_dir / f"hand_data_{side}.npz"))
            self._run_dynhamr_refinement(data_sub_folder, save_dir, fx, fy, cx, cy, T, H, W, frames_rgb)
            self._save_hand_overlay_video(frames_rgb, save_dir, (fx + fy) / 2.0, H, W, fps=fps)
            return

        if hand_mode == "dynhamr_only":
            raise FileNotFoundError(
                f"[dynhamr] Raw checkpoint not found in {save_dir}. Run hand_detect first."
            )

        predictor = self._wrapper._predictor
        if predictor is None:
            self._wrapper._ensure_loaded()
            predictor = self._wrapper._predictor
        sam3_path = self.cfg.get("sam3_hand_masks")
        if not sam3_path:
            sam3_path = os.path.join(data_sub_folder, "hand_masks", "hand_masks.npy")
        if not os.path.exists(sam3_path):
            logger.info("[hand] hand_masks not found, generating with SAM3...")
            self._generate_hand_masks(frames_rgb, sam3_path)
        sam3_masks = np.load(sam3_path) if os.path.exists(sam3_path) else None
        if self.cfg.get("dynhamr_required", False) and sam3_masks is None:
            raise RuntimeError("Full optimization requires SAM3 masks; mask generation failed")
        if sam3_masks is not None:
            logger.info(f"[hand] SAM3 masks: {sam3_masks.shape}")

        # Step 1: Per-frame detection with SAM3 + edge filter
        report_progress("WiLoR 逐帧预测", 0, T)
        all_dets = {}
        n_edge, n_sam3 = 0, 0
        for t in range(T):
            if t % 30 == 0:
                report_progress("WiLoR 逐帧预测", t, T)
            if t % 100 == 0:
                logger.info(f"[hand] detecting {t}/{T}...")
            img_bgr = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
            boxes, is_right, scores = predictor._detect_hands(img_bgr)
            if len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                is_right[i] = (boxes[i][0] + boxes[i][2]) / 2.0 >= W / 2.0
            run_fn = predictor._run_wilor if self.cfg.get("anyhand_backend", "wilor") == "wilor" else predictor._run_hamer
            hands = run_fn(img_bgr, boxes, is_right, scores)

            mask = sam3_masks[t] if sam3_masks is not None and t < len(sam3_masks) else None
            if sam3_masks is not None and (mask is None or not mask.any()):
                continue

            frame_dets = []
            for h in hands:
                jc = h.keypoints_3d + h.cam_t[None, :]
                pf = h.focal_length
                kp2d = np.zeros((21, 2))
                kp2d[:, 0] = jc[:, 0] / jc[:, 2] * pf + W / 2.0
                kp2d[:, 1] = jc[:, 1] / jc[:, 2] * pf + H / 2.0

                if mask is not None:
                    n_in_mask = sum(1 for k in range(21)
                                   if 0 <= int(kp2d[k, 0]) < W and 0 <= int(kp2d[k, 1]) < H
                                   and mask[int(kp2d[k, 1]), int(kp2d[k, 0])])
                    if n_in_mask / 21.0 < 0.2:
                        n_sam3 += 1
                        continue

                MARGIN = 10
                in_frame = sum(1 for k in range(21)
                               if MARGIN < kp2d[k, 0] < W - MARGIN and MARGIN < kp2d[k, 1] < H - MARGIN)
                if in_frame / 21.0 < 0.7:
                    n_edge += 1
                    continue

                frame_dets.append({
                    "hand": h, "joints_cam": jc, "kp2d": kp2d,
                    "projection_focal": pf, "score": h.score,
                    "wrist_u": kp2d[0, 0], "wrist_v": kp2d[0, 1],
                })
            if frame_dets:
                all_dets[t] = frame_dets

        logger.info(f"[hand] SAM3 filtered: {n_sam3}, edge filtered: {n_edge}")
        report_progress("WiLoR 逐帧预测", T, T)

        # Step 2: Score-based tracking
        report_progress("跨帧跟踪与异常过滤")
        best_t, best_sc = None, -1
        for t, dets in all_dets.items():
            if len(dets) >= 2:
                ds = sorted(dets, key=lambda d: d["score"], reverse=True)
                sc = ds[0]["score"] + ds[1]["score"]
                if sc > best_sc:
                    best_sc = sc
                    best_t = t

        left_r, right_r = {}, {}
        if best_t is not None:
            ds = sorted(all_dets[best_t], key=lambda d: d["wrist_u"])
            left_wrist = np.array([ds[0]["wrist_u"], ds[0]["wrist_v"]])
            right_wrist = np.array([ds[-1]["wrist_u"], ds[-1]["wrist_v"]])
            pack = lambda d: {
                "joints_cam": d["joints_cam"], "kp2d": d["kp2d"],
                "projection_focal": d["projection_focal"], "score": d["score"],
                "hand": d["hand"],
            }
            left_r[best_t] = pack(ds[0])
            right_r[best_t] = pack(ds[-1])

            for direction in ['forward', 'backward']:
                if direction == 'forward':
                    fr = range(best_t + 1, T)
                else:
                    fr = range(best_t - 1, -1, -1)
                    left_wrist = np.array([ds[0]["wrist_u"], ds[0]["wrist_v"]])
                    right_wrist = np.array([ds[-1]["wrist_u"], ds[-1]["wrist_v"]])
                for t in fr:
                    if t not in all_dets:
                        continue
                    for d in all_dets[t]:
                        w = np.array([d["wrist_u"], d["wrist_v"]])
                        dl = np.linalg.norm(w - left_wrist)
                        dr = np.linalg.norm(w - right_wrist)
                        p = pack(d)
                        if dl < dr:
                            if t not in left_r or d["score"] > left_r[t]["score"]:
                                left_r[t] = p
                        else:
                            if t not in right_r or d["score"] > right_r[t]["score"]:
                                right_r[t] = p
                    if t in left_r:
                        left_wrist = left_r[t]["kp2d"][0]
                    if t in right_r:
                        right_wrist = right_r[t]["kp2d"][0]

        logger.info(f"[hand] Tracked: L={len(left_r)}, R={len(right_r)}")

        # Step 3: Jump filter
        for res in [left_r, right_r]:
            fids = sorted(res.keys())
            if len(fids) < 3:
                continue
            w = np.array([res[f]["joints_cam"][0] for f in fids])
            v = np.linalg.norm(np.diff(w, axis=0), axis=1)
            med = np.median(v[v > 0]) if np.any(v > 0) else 0
            for j in np.where(v > max(med * 4, 0.003))[0]:
                res.pop(fids[j + 1], None)

        logger.info(f"[hand] After jump filter: L={len(left_r)}, R={len(right_r)}")

        # Step 4: Save hand_data_*.npz
        for hand_name, res in [("left", left_r), ("right", right_r)]:
            fids = sorted(res.keys())
            kpts_3d = np.zeros((T, 21, 3), dtype=np.float32)
            kpts_2d = np.zeros((T, 21, 2), dtype=np.float32)
            hand_detected = np.zeros(T, dtype=bool)
            eef_rot = np.zeros((T, 3, 3), dtype=np.float32)
            mano_orient, mano_pose, mano_betas, mano_transl = [], [], [], []
            projection_focals = []

            for f in fids:
                kpts_3d[f] = res[f]["joints_cam"]
                kpts_2d[f] = res[f]["kp2d"]
                hand_detected[f] = True
                h = res[f]["hand"]
                mano_orient.append(h.mano_pose[:3])
                mano_pose.append(h.mano_pose[3:48])
                mano_betas.append(h.mano_shape)
                mano_transl.append(h.cam_t)
                projection_focals.append(h.focal_length)

            # Save raw checkpoint (before DynHaMR) for fast re-postprocess
            np.savez_compressed(
                str(save_dir / f"hand_data_raw_{hand_name}.npz"),
                hand_detected=hand_detected, kpts_2d=kpts_2d, kpts_3d=kpts_3d,
                eef_rot=eef_rot, frame_indices=np.arange(T),
                mano_orient=mano_orient, mano_pose=mano_pose,
                mano_betas=mano_betas, mano_transl=mano_transl,
                detected_frame_ids=fids,
                projection_focals=projection_focals,
            )
            if hand_mode != "detect_only":
                np.savez_compressed(
                    str(save_dir / f"hand_data_{hand_name}.npz"),
                    hand_detected=hand_detected, kpts_2d=kpts_2d, kpts_3d=kpts_3d,
                    eef_rot=eef_rot, frame_indices=np.arange(T),
                    mano_orient=mano_orient, mano_pose=mano_pose,
                    mano_betas=mano_betas, mano_transl=mano_transl,
                    detected_frame_ids=fids,
                    projection_focals=projection_focals,
                )

        # Step 5: Save AnyHand overlay (before DynHaMR)
        if self.cfg.get("save_raw_overlay", True):
            self._save_hand_overlay_video(frames_rgb, save_dir, (fx + fy) / 2.0, H, W,
                                          suffix="_anyhand", data_prefix="hand_data_raw", fps=fps)

        if hand_mode == "detect_only":
            logger.info("[hand_detect] Detection complete, skipping DynHaMR")
            return

        # Step 6: DynHaMR temporal refinement
        self._run_dynhamr_refinement(data_sub_folder, save_dir, fx, fy, cx, cy, T, H, W, frames_rgb)

        # Step 7: Generate overlay (after DynHaMR)
        self._save_hand_overlay_video(frames_rgb, save_dir, (fx + fy) / 2.0, H, W, fps=fps)

    def _run_dynhamr_refinement(self, data_sub_folder, save_dir, fx, fy, cx, cy, T, H, W, frames_rgb):
        """Run DynHaMR temporal optimization on AnyHand per-frame results."""

        dynhamr_root = self.cfg.get("dynhamr_root")
        if not dynhamr_root or not os.path.isdir(dynhamr_root):
            if self.cfg.get("dynhamr_required", False):
                raise FileNotFoundError(f"Required DynHaMR repo not found: {dynhamr_root}")
            logger.warning(f"[dynhamr] dynhamr_root not found: {dynhamr_root}")
            return

        anyhand_focal = float(self.cfg.get("anyhand_focal", 2500))
        gt_focal = (fx + fy) / 2.0
        z_min = float(self.cfg.get("dynhamr_z_min", 0.15))
        z_max = float(self.cfg.get("dynhamr_z_max", 0.4))
        gap_fill = int(self.cfg.get("dynhamr_gap_fill", 3))

        seq_name = os.path.basename(data_sub_folder) or "demo"
        import hashlib
        parent_hash = hashlib.md5(str(data_sub_folder).encode()).hexdigest()[:6]
        seq_name = f"dh_{parent_hash}_{seq_name}"
        dh_test_dir = os.path.join(dynhamr_root, "test")

        logger.info(f"[dynhamr] Starting refinement: gt_focal={gt_focal:.1f}, anyhand_focal={anyhand_focal:.0f}")
        report_progress("准备 DynHaMR 输入")

        # Step 1: Convert to DynHaMR format
        self._prepare_dynhamr_images(frames_rgb, dh_test_dir, seq_name)
        self._convert_to_dynhamr(save_dir, dh_test_dir, seq_name, fx, fy, cx, cy,
                                 anyhand_focal, T, H, W)

        # Step 2: Run DynHaMR optimization
        report_progress("DynHaMR/HMP 完整优化", message="root → smooth → motion prior，迭代日志实时输出")
        self._hmp_fallback_reason = None
        success = self._run_dynhamr_subprocess(dynhamr_root, dh_test_dir, seq_name)
        if not success:
            hmp_reason = self._hmp_fallback_reason
            quality = None
            if hmp_reason:
                quality = self._try_smooth_fallback(
                    dynhamr_root, dh_test_dir, seq_name, save_dir,
                    fx, fy, cx, cy, T, H, W, gap_fill, z_min, z_max)
                if quality["accepted"]:
                    self.optimization_status = {
                        "succeeded": True, "fallback": True, "stage": "hmp",
                        "result_source": "smooth_fit", "root_smooth_completed": True,
                        "prior_executed": False, "hmp_succeeded": False,
                        "hmp_skipped": getattr(self, "_hmp_skipped", False),
                        "reason": hmp_reason, "quality": quality,
                    }
                    (Path(data_sub_folder) / "optimization_status.json").write_text(
                        json.dumps(self.optimization_status, ensure_ascii=False, indent=2))
                    report_progress("smooth 检查通过，使用优化结果生成视频")
                    return
            if not hmp_reason and self.cfg.get("dynhamr_required", False):
                raise RuntimeError("DynHaMR optimization failed; see dynhamr_stdout.log and dynhamr_stderr.log in the demo directory")
            self.optimization_status = {"succeeded": False, "fallback": True, "stage": "hmp" if hmp_reason else "dynhamr",
                                        "result_source": "wilor_filtered", "reason": hmp_reason or "DynHaMR optimization failed"}
            if quality is not None:
                self.optimization_status.update(quality=quality, prior_executed=False,
                                                root_smooth_completed=True, hmp_succeeded=False,
                                                hmp_skipped=getattr(self, "_hmp_skipped", False))
            (Path(data_sub_folder) / "optimization_status.json").write_text(json.dumps(self.optimization_status, ensure_ascii=False, indent=2))
            logger.warning("[dynhamr] %s; keeping WiLoR predictions and exporting video", self.optimization_status["reason"])
            report_progress("优化失败回退，继续生成视频", message=self.optimization_status["reason"])
            return

        # Step 3: Post-process and write back
        report_progress("优化结果 MANO 后处理")
        self._postprocess_dynhamr(dynhamr_root, dh_test_dir, seq_name, save_dir,
                                  fx, fy, cx, cy, T, gap_fill, z_min, z_max)
        self.optimization_status = {"succeeded": True, "fallback": False, "result_source": "hmp"}
        (Path(data_sub_folder) / "optimization_status.json").write_text(json.dumps(self.optimization_status, indent=2))

    def _try_smooth_fallback(self, dynhamr_root, dh_test_dir, seq_name, save_dir,
                             fx, fy, cx, cy, T, H, W, gap_fill, z_min, z_max):
        """Validate a completed smooth result before replacing any WiLoR files."""
        from ego2robot.utils.optimization_quality import assess_smooth_result
        report_progress("smooth 结果质量检查")
        try:
            source = getattr(self, "_smooth_result_path", None)
            if not source:
                raise FileNotFoundError("this run did not record a completed smooth result")
            # The subprocess supplies its exact completed stage, including iteration.
            # Never select an initial or unrelated result from another day's cache.
            if Path(source).parent.name != "smooth_fit" or not Path(source).is_file():
                raise FileNotFoundError(f"invalid smooth result: {source}")
            baseline = {}
            for side in ("left", "right"):
                with np.load(save_dir / f"hand_data_{side}.npz", allow_pickle=True) as data:
                    baseline[side] = dict(data)
            candidate = self._postprocess_dynhamr(
                dynhamr_root, dh_test_dir, seq_name, save_dir,
                fx, fy, cx, cy, T, gap_fill, z_min, z_max,
                result_path=source, write_output=False)
            quality = assess_smooth_result(baseline, candidate, T, W, H, self.cfg)
            quality["source_file"] = str(source)
            if quality["accepted"]:
                self._write_dynhamr_hands(save_dir, candidate)
            else:
                logger.warning("[dynhamr] smooth rejected: %s", "; ".join(quality["reasons"]))
            return quality
        except Exception as error:
            logger.warning("[dynhamr] smooth unavailable or invalid: %s", error)
            return {"accepted": False, "reasons": [f"{type(error).__name__}: {error}"]}

    def _prepare_dynhamr_images(self, frames_rgb, dh_test_dir, seq_name):
        """Save video frames as JPGs for DynHaMR."""
        import cv2
        img_dir = os.path.join(dh_test_dir, "images", seq_name)
        os.makedirs(img_dir, exist_ok=True)
        for i, frame in enumerate(frames_rgb):
            cv2.imwrite(os.path.join(img_dir, f"{i:06d}.jpg"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return img_dir

    def _convert_to_dynhamr(self, save_dir, dh_test_dir, seq_name,
                             fx, fy, cx, cy, anyhand_focal, T, H, W):
        """Convert hand_data_*.npz to DynHaMR track_preds format."""
        import shutil

        gt_focal = (fx + fy) / 2.0
        track_dir = os.path.join(dh_test_dir, "dynhamr", "track_preds", seq_name)
        cam_dir = os.path.join(dh_test_dir, "dynhamr", "cameras", seq_name, "shot-0")
        shot_dir = os.path.join(dh_test_dir, "dynhamr", "shot_idcs")

        for tid in ["000", "001"]:
            d = os.path.join(track_dir, tid)
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)
        os.makedirs(cam_dir, exist_ok=True)
        os.makedirs(shot_dir, exist_ok=True)

        for side, tid in [("left", "000"), ("right", "001")]:
            data = np.load(str(save_dir / f"hand_data_{side}.npz"), allow_pickle=True)
            det_ids = data['detected_frame_ids']
            if len(det_ids) == 0:
                continue

            mano_orient = data['mano_orient']
            mano_pose = data['mano_pose']
            mano_betas = data['mano_betas']
            mano_transl = data['mano_transl']
            kpts_2d = data['kpts_2d']
            is_left = (side == "left")

            for i, fid in enumerate(det_ids):
                fid = int(fid)
                if i >= len(mano_orient):
                    break

                cam_t = np.array(mano_transl[i], dtype=np.float64)
                if is_left:
                    cam_t[0] *= -1
                cam_t[2] *= gt_focal / anyhand_focal

                mano_json = {
                    "betas": np.array(mano_betas[i]).tolist(),
                    "body_pose": np.array(mano_pose[i]).reshape(15, 3).tolist(),
                    "global_orient": np.array(mano_orient[i]).tolist(),
                    "cam_trans": cam_t.tolist(),
                    "is_right": int(not is_left),
                }
                frame_str = f"{fid:06d}"
                with open(os.path.join(track_dir, tid, f"{frame_str}_mano.json"), "w") as f:
                    json.dump(mano_json, f)

                kp3 = np.zeros((21, 3))
                kp3[:, :2] = kpts_2d[fid]
                kp3[:, 2] = 1.0
                kp_json = {"people": [{"pose_keypoints_2d": kp3.flatten().tolist()}]}
                with open(os.path.join(track_dir, tid, f"{frame_str}_keypoints.json"), "w") as f:
                    json.dump(kp_json, f)

        # cameras.npz with GT focal
        w2c = np.tile(np.eye(4), (T, 1, 1))
        cam_data = {
            "w2c": w2c,
            "focal": np.float64(gt_focal),
            "intrins": np.tile(np.array([fx, fy, cx, cy]), (T, 1)),
            "width": np.int64(W),
            "height": np.int64(H),
        }
        np.savez(os.path.join(cam_dir, "cameras.npz"), **cam_data)

        shots = {f"{i:06d}.jpg": 0 for i in range(T)}
        with open(os.path.join(shot_dir, f"{seq_name}.json"), "w") as f:
            json.dump(shots, f)

        logger.info(f"[dynhamr] Converted to DynHaMR format: {track_dir}")

    def _run_dynhamr_subprocess(self, dynhamr_root, dh_test_dir, seq_name):
        """Run DynHaMR optimization via subprocess."""
        import subprocess as sp

        dyn_dir = os.path.join(dynhamr_root, "dyn-hamr")
        data_dir = os.path.join(dynhamr_root, "_DATA", "data")

        # A subprocess isolates module names; it uses the same environment by default.
        dynhamr_python = (self.cfg.get("dynhamr_python")
                          or os.environ.get("DYNHAMR_PYTHON") or sys.executable)
        launcher = Path(__file__).resolve().parents[2] / "scripts" / "run_dynhamr.py"

        cmd = [
            dynhamr_python, str(launcher), "--dynhamr-root", str(dynhamr_root),
            "data=video_vipe",
            "run_opt=True", "run_prior=True", "run_vis=False",
            # Hydra parses each argv entry itself; quote Chinese names and paths
            # even though subprocess already passes arguments without a shell.
            f"data.seq={json.dumps(seq_name, ensure_ascii=False)}",
            f"data.root={json.dumps(dh_test_dir, ensure_ascii=False)}",
            "is_static=True",
            f"paths.DATA_DIR={json.dumps(data_dir, ensure_ascii=False)}",
            "overwrite=True",
        ]

        logger.info(f"[dynhamr] Running: {' '.join(cmd[:6])}...")
        cmd.append(f"gpu={int(self.cfg.get('dynhamr_gpu', 0))}")
        log_dir = Path(self.cfg.get("dynhamr_log_dir", dh_test_dir))
        log_dir.mkdir(parents=True, exist_ok=True)
        hmp_status = log_dir / "hmp_fallback.json"
        hmp_status.unlink(missing_ok=True)
        (log_dir / "smooth_completed.json").unlink(missing_ok=True)
        hmp_running = log_dir / "hmp_running.json"
        hmp_running.unlink(missing_ok=True)
        self._hmp_fallback_reason = None
        self._smooth_result_path = None
        self._hmp_skipped = False
        cmd += ["--hmp-fallback-status", str(hmp_status.resolve())]
        try:
            result = run_streamed_command(
                cmd, cwd=dyn_dir, stdout_path=log_dir / "dynhamr_stdout.log",
                stderr_path=log_dir / "dynhamr_stderr.log",
                timeout=int(self.cfg.get("dynhamr_timeout", 7200)),
                progress_only=True,
                stage_markers=(("start optimization", "DynHaMR root 优化"),
                               ("root optimization time", "DynHaMR smooth 优化"),
                               ("smooth optimization time", "HMP 运动先验优化"),
                               ("prior optimization time", "DynHaMR/HMP 结果保存")))
            if result.returncode != 0:
                if hmp_running.is_file():
                    self._smooth_result_path = json.loads(hmp_running.read_text()).get("smooth_result")
                    self._hmp_fallback_reason = f"HMP worker exited with code {result.returncode}: {result.stderr[-500:]}"
                    return False
                logger.error(f"[dynhamr] Failed: {result.stderr[-500:]}")
                self._cleanup_dynhamr_output(dynhamr_root, seq_name)
                return False
            if hmp_status.is_file():
                status = json.loads(hmp_status.read_text())
                if status.get("stage") == "hmp" and status.get("fallback"):
                    self._smooth_result_path = status.get("smooth_result")
                    self._hmp_skipped = status.get("skipped", False)
                    self._hmp_fallback_reason = status["reason"]
                    return False
            logger.info("[dynhamr] Optimization completed")
            return True
        except sp.TimeoutExpired:
            if hmp_running.is_file():
                self._smooth_result_path = json.loads(hmp_running.read_text()).get("smooth_result")
                self._hmp_fallback_reason = "HMP worker timed out"
                return False
            logger.error("[dynhamr] Timeout")
            self._cleanup_dynhamr_output(dynhamr_root, seq_name)
            return False

    def _cleanup_dynhamr_output(self, dynhamr_root, seq_name):
        """Remove partial DynHaMR output to avoid stale cache on retry."""
        import glob, shutil
        pattern = os.path.join(dynhamr_root, "outputs", "logs", "video-custom", "*", f"{seq_name}-*")
        for d in glob.glob(pattern):
            shutil.rmtree(d, ignore_errors=True)
            logger.info(f"[dynhamr] Cleaned partial output: {d}")

    def _postprocess_dynhamr(self, dynhamr_root, dh_test_dir, seq_name, save_dir,
                              fx, fy, cx, cy, T, gap_fill, z_min, z_max,
                              *, result_path=None, write_output=True):
        """Build hand arrays from prior or smooth, optionally publishing them."""
        import torch
        import sys
        import glob
        from scipy.spatial.transform import Rotation

        sys.path.insert(0, os.path.join(dynhamr_root, "dyn-hamr"))
        from ego2robot.utils.legacy_mano import prepare_mano_loading
        prepare_mano_loading()
        from body_model import MANO

        # Find output npz
        pattern = os.path.join(dynhamr_root, "outputs", "logs", "video-custom", "*",
                               f"{seq_name}-*", "prior", f"{seq_name}_000000_world_results.npz")
        npz_files = [str(result_path)] if result_path is not None else sorted(glob.glob(pattern))
        if not npz_files:
            if self.cfg.get("dynhamr_required", False):
                raise FileNotFoundError(f"DynHaMR did not produce motion-prior output: {pattern}")
            logger.warning(f"[dynhamr] No output found: {pattern}")
            return {}
        npz_path = npz_files[-1]
        logger.info(f"[dynhamr] Loading {npz_path}")

        with np.load(npz_path) as npz:
            trans, orient, pose, betas, is_right = (npz[key] for key in
                ('trans', 'root_orient', 'pose_body', 'betas', 'is_right'))
        if trans.ndim != 3 or trans.shape[-1] != 3:
            raise ValueError("invalid optimized translation shape")
        N_hands, T_dh = trans.shape[:2]
        if (not 1 <= N_hands <= 2 or not 1 <= T_dh <= T
                or orient.shape != (N_hands, T_dh, 3)
                or pose.shape[:2] != (N_hands, T_dh) or pose.size != N_hands * T_dh * 45
                or betas.shape != (N_hands, 10) or is_right.shape != (N_hands, T_dh)
                or not all(np.isfinite(a).all() for a in (trans, orient, pose, betas, is_right))):
            raise ValueError("invalid optimized MANO arrays")
        if (not np.isin(is_right, [0, 1]).all()
                or not (is_right == is_right[:, :1]).all()
                or len(np.unique(is_right[:, 0])) != N_hands):
            raise ValueError("invalid optimized hand identities")

        # Find start frame from track_preds
        track_dir = os.path.join(dh_test_dir, "dynhamr", "track_preds", seq_name)
        all_frames = set()
        det_per_tid = {}
        for tid in sorted(os.listdir(track_dir)):
            td = os.path.join(track_dir, tid)
            if not os.path.isdir(td):
                continue
            frames = set(int(f.split('_')[0]) for f in os.listdir(td) if '_mano.json' in f)
            det_per_tid[tid] = frames
            all_frames.update(frames)
        start_frame = min(all_frames) if all_frames else 0
        if start_frame < 0 or start_frame + T_dh > T:
            raise ValueError("optimized frames do not fit the video timeline")

        # Build detection masks
        mano_model = MANO(
            batch_size=1, pose2rot=True,
            model_path=os.path.join(dynhamr_root, "_DATA", "data", "mano"),
            is_rhand=True
        )

        updated_hands = {}
        for h in range(N_hands):
            hand_right = bool(is_right[h, 0] > 0.5)
            side = "right" if hand_right else "left"
            tid = "001" if hand_right else "000"
            det_frames = det_per_tid.get(tid, set())

            mask = np.zeros(T_dh, dtype=bool)
            for t in range(T_dh):
                if start_frame + t in det_frames:
                    mask[t] = True

            # Fill small gaps
            det_idx = np.where(mask)[0]
            if len(det_idx) > 1:
                for i in range(len(det_idx) - 1):
                    if det_idx[i + 1] - det_idx[i] - 1 <= gap_fill:
                        for t in range(det_idx[i] + 1, det_idx[i + 1]):
                            mask[t] = True

            # Trim boundary frames (smoothing from 0 → detection is unreliable)
            TRIM = 3
            det_idx = np.where(mask)[0]
            if len(det_idx) > 2 * TRIM:
                for t in det_idx[:TRIM]:
                    mask[t] = False
                for t in det_idx[-TRIM:]:
                    mask[t] = False

            # MANO forward + z filter + build output arrays
            kpts_3d = np.zeros((T, 21, 3), dtype=np.float32)
            kpts_2d = np.zeros((T, 21, 2), dtype=np.float32)
            hand_detected = np.zeros(T, dtype=bool)
            eef_rot = np.zeros((T, 3, 3), dtype=np.float32)

            for t in range(T_dh):
                if not mask[t]:
                    continue
                orig_f = start_frame + t
                if orig_f >= T:
                    continue

                tr = torch.tensor(trans[h, t:t+1], dtype=torch.float32)
                ori = torch.tensor(orient[h, t:t+1], dtype=torch.float32)
                po = torch.tensor(pose[h, t:t+1].reshape(1, 45), dtype=torch.float32)
                be = torch.tensor(betas[h:h+1], dtype=torch.float32)

                with torch.no_grad():
                    out = mano_model(global_orient=ori, hand_pose=po, betas=be, transl=tr)
                j3d = out.joints[0, :21].numpy()

                if not hand_right:
                    j3d[:, 0] = -j3d[:, 0]

                if not np.isfinite(j3d).all() or (j3d[:, 2] <= 0).any():
                    continue
                wrist_z = j3d[0, 2]
                if wrist_z < z_min or wrist_z > z_max:
                    continue

                kpts_3d[orig_f] = j3d
                kpts_2d[orig_f, :, 0] = j3d[:, 0] / j3d[:, 2] * fx + cx
                kpts_2d[orig_f, :, 1] = j3d[:, 1] / j3d[:, 2] * fy + cy
                hand_detected[orig_f] = True
                eef_rot[orig_f] = Rotation.from_rotvec(orient[h, t]).as_matrix()

            det_ids = np.where(hand_detected)[0]
            logger.info(f"[dynhamr] {side}: {len(det_ids)} frames after post-processing")

            # Overwrite hand_data npz
            with np.load(save_dir / f"hand_data_{side}.npz", allow_pickle=True) as previous:
                old = dict(previous)
            old['kpts_3d'] = kpts_3d
            old['kpts_2d'] = kpts_2d
            old['hand_detected'] = hand_detected
            old['eef_rot'] = eef_rot
            old['detected_frame_ids'] = det_ids
            updated_hands[side] = old

        if write_output:
            self._write_dynhamr_hands(save_dir, updated_hands)
        return updated_hands

    @staticmethod
    def _write_dynhamr_hands(save_dir, hands):
        """Serialize all hands before replacing published files."""
        import tempfile
        with tempfile.TemporaryDirectory(prefix=".dynhamr-", dir=save_dir) as temporary:
            for side, data in hands.items():
                np.savez_compressed(Path(temporary) / f"hand_data_{side}.npz", **data)
            for side in hands:
                name = f"hand_data_{side}.npz"
                os.replace(Path(temporary) / name, save_dir / name)
        logger.info("[dynhamr] Post-processing done, hand_data_*.npz updated")

    def _generate_hand_masks(self, frames_rgb, save_path):
        """Generate hand masks using SAM3 in a subprocess to avoid dtype contamination."""
        import subprocess as sp
        import sys
        import tempfile

        frames_rgb = np.asarray(frames_rgb)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as f:
            frames_path = f.name
            np.save(f, frames_rgb)

        ckpt = self.cfg.get("sam3_checkpoint", "")
        script = f"""
import numpy as np
from PIL import Image as PILImage
from sam3.model_builder import build_sam3_video_predictor
from ego2robot.utils.progress import report_progress
import cv2

frames_rgb = np.load({frames_path!r})
T, H, W = frames_rgb.shape[:3]
ckpt = {ckpt!r}
report_progress("SAM3 手部掩码", 0, T, "加载模型")
predictor = build_sam3_video_predictor(checkpoint_path=ckpt, gpus_to_use=[0]) if ckpt else build_sam3_video_predictor(gpus_to_use=[0])
pil_frames = [PILImage.fromarray(frames_rgb[i]) for i in range(T)]
resp = predictor.handle_request(dict(type="start_session", resource_path=pil_frames))
sid = resp["session_id"]
try:
    resp = predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=T//2, text="hand"))
    n = len(resp["outputs"].get("out_obj_ids", []))
    print(f"SAM3 hand mask: frame {{T//2}} detected {{n}} objects")
    if n == 0:
        for ff in [0, T//4, 3*T//4, T-1]:
            resp = predictor.handle_request(dict(type="add_prompt", session_id=sid, frame_index=ff, text="hand"))
            if len(resp["outputs"].get("out_obj_ids", [])) > 0: break
    masks = np.zeros((T, H, W), dtype=bool)
    seen = set()
    for resp in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=sid)):
        fi = resp["frame_index"]
        seen.add(fi)
        if len(seen) % 30 == 0 or len(seen) == T:
            report_progress("SAM3 手部掩码", len(seen), T)
        om = resp["outputs"].get("out_binary_masks")
        if om is not None and hasattr(om, 'any'):
            m = om.any(dim=0).cpu().numpy() if hasattr(om, 'cpu') else np.asarray(om).any(axis=0)
            if m.shape != (H, W): m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            masks[fi] = m
finally:
    predictor.handle_request(dict(type="close_session", session_id=sid))
np.save({save_path!r}, masks)
print(f"Saved: {{masks.any(axis=(1,2)).sum()}}/{{T}} frames")
"""
        logger.info("[hand] Generating SAM3 hand masks in subprocess...")
        report_progress("SAM3 手部掩码", 0, len(frames_rgb))
        try:
            result = run_streamed_command(
                [sys.executable, "-u", "-c", script],
                stdout_path=Path(save_path).parent / "sam3_stdout.log",
                stderr_path=Path(save_path).parent / "sam3_stderr.log", timeout=1800)
        finally:
            os.unlink(frames_path)

        if result.returncode != 0:
            if self.cfg.get("dynhamr_required", False):
                raise RuntimeError(f"SAM3 mask generation failed: {result.stderr[-2000:]}")
            logger.error(f"[hand] SAM3 subprocess failed: {result.stderr[-500:]}")


class HandDetectProcessor(HandProcessor):
    """AnyHand detection only (GPU). Saves raw checkpoint, skips DynHaMR."""

    def __init__(self, args):
        from omegaconf import OmegaConf
        args = OmegaConf.create({**OmegaConf.to_container(args, resolve=True), "hand_mode": "detect_only"})
        super().__init__(args)

    @staticmethod
    def check_exists(data_sub_folder, cfg=None):
        save_dir = os.path.join(data_sub_folder, "hand_processor")
        return (os.path.exists(os.path.join(save_dir, "hand_data_raw_left.npz")) and
                os.path.exists(os.path.join(save_dir, "hand_data_raw_right.npz")))


class DynHaMRProcessor(HandProcessor):
    """DynHaMR optimization only (CPU). Reads raw checkpoint from hand_detect."""

    def __init__(self, args):
        from omegaconf import OmegaConf
        args = OmegaConf.create({**OmegaConf.to_container(args, resolve=True), "hand_mode": "dynhamr_only"})
        super().__init__(args)

    @staticmethod
    def check_exists(data_sub_folder, cfg=None):
        save_dir = os.path.join(data_sub_folder, "hand_processor")
        return (os.path.exists(os.path.join(save_dir, "hand_data_left.npz")) and
                os.path.exists(os.path.join(save_dir, "hand_data_right.npz")) and
                os.path.exists(os.path.join(save_dir, "video_hand_overlay.mp4")))

    def close(self):
        if self._wrapper is not None:
            self._wrapper.release()
            self._wrapper = None
