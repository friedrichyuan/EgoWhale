"""DepthPose Processor — Stage 1: DA3 depth + pose + intrinsics.

Replaces MoGe-2 + DROID-SLAM with a single DA3 forward pass.
Outputs depth maps, camera extrinsics (c2w), and intrinsics to disk.
"""

import os
import json
import logging
import numpy as np
import cv2
from pathlib import Path

from omegaconf import DictConfig

from ego2robot.processors.base_processor import BaseProcessor, register_processor
from ego2robot.utils.progress import report_progress

logger = logging.getLogger(__name__)


def da3_frame_indices(count, stride=1, cap=0):
    if count < 1 or stride < 1 or cap < 0 or cap == 1:
        raise ValueError("Invalid DA3 sampling budget")
    indices = list(range(0, count, stride))
    if indices[-1] != count - 1:
        indices.append(count - 1)
    if cap and len(indices) > cap:
        indices = [indices[i] for i in np.linspace(0, len(indices) - 1, cap).round().astype(int)]
    return indices


@register_processor("depth_pose")
class DepthPoseProcessor(BaseProcessor):
    """Estimate depth, camera pose, and intrinsics using DA3.

    Input:  video frames (RGB) in ``data_sub_folder/``
    Output: ``data_sub_folder/``
        - depth/000000.npy, 000001.npy, ...  (per-frame depth maps)
        - cam_extrinsics.npz   (R_c2w, t_c2w, frame_indices)
        - cam_intrinsics.json  (K, hfov, vfov, focal, fps)
    """

    gpu_fraction: float = 0.5

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._model = None  # lazy

    def _ensure_model(self):
        if self._model is not None:
            return
        from ego2robot.models.da3_wrapper import DA3Wrapper, DA3_DEFAULT_MODEL_ROOT
        self._model = DA3Wrapper(
            model_name=self.cfg.get("depth_model", "da3nested"),
            device="cuda:0",
            model_root=self.cfg.get("da3_model_root", DA3_DEFAULT_MODEL_ROOT),
        )

    def process_one_demo(self, data_sub_folder: str) -> None:
        if isinstance(data_sub_folder, bytes):
            data_sub_folder = data_sub_folder.decode('utf-8')
        save_dir = Path(data_sub_folder)
        save_dir.mkdir(parents=True, exist_ok=True)

        # ── Standardize video filename ───────────────────
        self._ensure_standard_video(data_sub_folder)

        stride = int(self.cfg.get("depth_stride", 1))
        cap = int(self.cfg.get("da3_max_views", 0))
        prepared = save_dir / "input.json"
        if prepared.is_file() and (save_dir / "video_L.mp4").is_file():
            import av
            info = json.loads(prepared.read_text())
            T, fps = info["frames"], info["fps"]
            W, H = info["size"]
            indices = da3_frame_indices(T, stride, cap)
            selected = set(indices)
            with av.open(str(save_dir / "video_L.mp4")) as video:
                frames = [frame.to_ndarray(format="rgb24") for i, frame in enumerate(video.decode(video=0))
                          if i in selected]
            if len(frames) != len(indices):
                raise ValueError("DA3 采样帧与准备阶段不一致")
        else:
            frames, fps = self._load_frames(data_sub_folder)
            if not frames:
                raise ValueError(f"No video frames found in {data_sub_folder}")
            T, (H, W) = len(frames), frames[0].shape[:2]
            indices = da3_frame_indices(T, stride, cap)
            frames = [frames[i] for i in indices]

        stride_indices = indices
        report_progress("DA3 相机估计", message=f"推理 {len(frames)}/{T} 帧")
        self._ensure_model()
        result = self._model.predict(frames, process_res=self.cfg.get("da3_process_res", 504))
        (save_dir / "da3_samples.json").write_text(json.dumps({"indices": indices, "max_views": cap}))
        depth_dir = save_dir / "depth"
        depth_dir.mkdir(exist_ok=True)
        if self.cfg.get("save_depth_maps", True):
            depths = [cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                      if d.shape != (H, W) else d for d in result["depth"]]
            bulk = np.lib.format.open_memmap(depth_dir / "all_depths.npy", mode="w+",
                                            dtype=np.float32, shape=(T, H, W))
            j = 0
            for i in range(T):
                if j + 1 < len(indices) and i == indices[j + 1]:
                    j += 1
                if i == indices[j]:
                    depth = depths[j]
                else:
                    alpha = (i - indices[j]) / (indices[j + 1] - indices[j])
                    depth = depths[j] * (1 - alpha) + depths[j + 1] * alpha
                bulk[i] = depth
                np.save(depth_dir / f"{i:06d}.npy", bulk[i])
            del bulk
        report_progress("DA3 相机参数保存", message="保存估计的内参与位姿")

        # ── Save camera extrinsics ───────────────────────
        ext_path = save_dir / "cam_extrinsics.npz"
        known_intrinsics = self.cfg.get("known_intrinsics", False)

        if known_intrinsics and ext_path.exists():
            logger.info("[depth_pose] known_intrinsics: keeping existing GT extrinsics (not overwritten)")
        elif known_intrinsics and self.calibration is not None and "head_T" in self.calibration:
            head_T = self.calibration["head_T"]
            R_c2w = np.tile(head_T[:3, :3], (T, 1, 1))
            t_c2w = np.tile(head_T[:3, 3], (T, 1))
            R_w2c = np.tile(head_T[:3, :3].T, (T, 1, 1))
            t_w2c = np.tile(-head_T[:3, :3].T @ head_T[:3, 3], (T, 1))
            np.savez_compressed(
                str(ext_path),
                R_c2w=R_c2w, t_c2w=t_c2w,
                R_w2c=R_w2c, t_w2c=t_w2c,
                frame_indices=np.arange(T),
            )
            logger.info("[depth_pose] known_intrinsics: wrote GT extrinsics from calibration to %s", ext_path)
        elif result["c2w"] is not None:
            R_c2w = np.stack([c[:3, :3] for c in result["c2w"]])
            t_c2w = np.stack([c[:3, 3] for c in result["c2w"]])
            if len(stride_indices) != T:
                from scipy.spatial.transform import Rotation, Slerp
                R_c2w = Slerp(stride_indices, Rotation.from_matrix(R_c2w))(np.arange(T)).as_matrix()
                t_c2w = np.stack([
                    np.interp(np.arange(T), stride_indices, t_c2w[:, axis])
                    for axis in range(3)
                ], axis=1)
            R_w2c = R_c2w.transpose(0, 2, 1)
            t_w2c = -np.einsum("tij,tj->ti", R_w2c, t_c2w)
            np.savez_compressed(
                str(save_dir / "cam_extrinsics.npz"),
                R_c2w=R_c2w,
                t_c2w=t_c2w,
                R_w2c=R_w2c,
                t_w2c=t_w2c,
                frame_indices=np.arange(T),
            )
        else:
            raise ValueError("DA3 did not return camera poses and no known extrinsics were supplied")

        # ── Save camera intrinsics ───────────────────────
        intr_path = save_dir / "cam_intrinsics.json"

        if known_intrinsics and self.calibration is not None and "intrinsics" in self.calibration:
            import math
            ci = self.calibration["intrinsics"]
            fx, fy, cx, cy = ci["fx"], ci["fy"], ci["cx"], ci["cy"]
            cw, ch = ci.get("width", W), ci.get("height", H)
            v_fov = 2 * math.degrees(math.atan(ch / (2 * fy)))
            intrinsics_dict = {
                "fps": fps, "width": cw, "height": ch,
                "hfov": 2 * math.atan(cw / (2 * fx)),
                "vfov": 2 * math.atan(ch / (2 * fy)),
                "focal": (fx + fy) / 2,
                "is_metric": 1,
                "left": {
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "new_w": cw, "new_h": ch, "orig_w": cw, "orig_h": ch,
                    "v_fov": v_fov, "fps": fps,
                },
            }
            with open(intr_path, "w") as f:
                json.dump(intrinsics_dict, f, indent=2)
            logger.info("[depth_pose] known_intrinsics: wrote GT intrinsics from calibration to %s", intr_path)
        elif known_intrinsics and intr_path.exists():
            logger.info("[depth_pose] known_intrinsics=True, keeping existing GT K (not overwritten)")
        else:
            K = result.get("intrinsics_orig")
            intrinsics_dict = {
                "fps": fps,
                "width": W,
                "height": H,
                "hfov": result.get("hfov"),
                "vfov": result.get("vfov"),
                "focal": result.get("focal"),
                "is_metric": int(result.get("is_metric", 0)),
            }
            if K is not None:
                import math
                intrinsics_dict["left"] = {
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "new_w": W,
                    "new_h": H,
                    "orig_w": W,
                    "orig_h": H,
                    "v_fov": math.degrees(result.get("vfov", 0)),
                    "fps": fps,
                }
            with open(intr_path, "w") as f:
                json.dump(intrinsics_dict, f, indent=2)

        logger.info(f"[depth_pose] saved to {save_dir}")

    @staticmethod
    def _ensure_standard_video(data_sub_folder: str):
        """Standardize video files for downstream processors.

        Creates:
        - video_rgb_imgs.mkv (symlink to source, for SAM3/segmentation)
        - video_L.mp4 (H.264 transcode, viewable in VSCode/browser)
        """
        import subprocess
        demo = Path(data_sub_folder)
        # Find any video file
        src_video = None
        for f in sorted(demo.iterdir()):
            if (f.suffix in (".mp4", ".mkv", ".avi") and not f.name.startswith(".")
                    and f.name != "video_annot.mp4"):
                src_video = f
                break
        if src_video is None:
            return

        # Symlink mkv (for processors that read raw frames)
        target_mkv = demo / "video_rgb_imgs.mkv"
        if not target_mkv.exists():
            os.symlink(src_video.name, str(target_mkv))

        # Transcode to H.264 MP4 (viewable in VSCode/browser)
        target_mp4 = demo / "video_L.mp4"
        if not target_mp4.exists():
            partial = demo / ".video_L.partial.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(src_video), "-c:v", "libx264",
                 "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(partial)],
                capture_output=True, check=True,
            )
            partial.replace(target_mp4)
            logger.info(f"[depth_pose] Transcoded {src_video.name} -> video_L.mp4 (H.264)")

    def _load_frames(self, data_sub_folder: str):
        """Load video frames from the demo directory.

        Looks for: video_left.mp4, video_rgb_imgs.mkv, *.mp4 (first found).
        Returns (list_of_rgb_arrays, fps).
        """
        import av

        demo = data_sub_folder
        if isinstance(demo, bytes):
            demo = demo.decode('utf-8')
        # Search for video file
        candidates = [
            os.path.join(demo, "video_left.mp4"),
            os.path.join(demo, "video_rgb_imgs.mkv"),
        ]
        # Also check any .mp4 in root
        for f in sorted(os.listdir(demo)):
            if isinstance(f, bytes):
                f = f.decode('utf-8', errors='replace')
            if f.endswith((".mp4", ".mkv")) and os.path.isfile(os.path.join(demo, f)):
                candidates.append(os.path.join(demo, f))

        video_path = None
        for c in candidates:
            if os.path.exists(c):
                video_path = c
                break

        if video_path is None:
            return None, 30

        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0

        frames = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
        container.close()

        if self.cfg.get("frame_sampling_method", "all") != "all":
            raise ValueError("Raw-video depth must retain every source frame; use depth_stride for interpolation")

        return frames, fps

    @staticmethod
    def check_exists(data_sub_folder: str, cfg=None) -> bool:
        has_intrinsics = (
            os.path.exists(os.path.join(data_sub_folder, "cam_extrinsics.npz"))
            and os.path.exists(os.path.join(data_sub_folder, "cam_intrinsics.json"))
        )
        has_depth = os.path.isdir(os.path.join(data_sub_folder, "depth"))
        # When known_intrinsics=True, base stage already wrote intrinsics/extrinsics,
        # but we still need DA3 to produce depth maps.
        return has_intrinsics and has_depth

    def close(self):
        if self._model is not None:
            self._model.release()
            self._model = None
