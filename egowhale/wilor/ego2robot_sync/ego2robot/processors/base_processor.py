import os
import json
import logging
import numpy as np
import shutil
import errno
import cv2
from typing import Tuple, Optional, List, Dict
from pathlib import Path
from omegaconf import DictConfig

from glob import glob
from natsort import natsorted
import numpy.typing as npt

from ego2robot.utils.image_utils import get_intrinsics_from_json, get_transformation_matrix_from_extrinsics
from ego2robot.processors.paths import Paths

logger = logging.getLogger(__name__)


def _get_project_folder() -> str:
    """Return the ego2robot project root (where configs/ lives)."""
    return str(Path(__file__).resolve().parent.parent.parent)

# ---------------------------------------------------------------------------
# Hand-side colour palette used by both processors
# ---------------------------------------------------------------------------
_HAND_COLORS: Dict[str, Tuple[int, int, int]] = {
    "left": (0, 128, 255),   # orange-ish in BGR
    "right": (255, 128, 0),  # blue-ish in BGR
}


# ── ego2robot extensions: registry + chunk helpers ──────────
PROCESSOR_REGISTRY: dict = {}

def register_processor(name: str):
    """Decorator that registers a processor class under *name*."""
    def _decorator(cls):
        PROCESSOR_REGISTRY[name] = cls
        cls.name = name
        return cls
    return _decorator

def enumerate_demos(processed_root: str, chunk_size: int = 1000):
    root = Path(processed_root)
    if not root.exists():
        return []
    demos = []
    for entry in sorted(root.iterdir()):
        if entry.name.startswith("chunk-") and entry.is_dir():
            for sub in sorted(entry.iterdir()):
                if sub.is_dir() and sub.name.isdigit():
                    demos.append(sub)
        elif entry.is_dir() and entry.name.isdigit():
            demos.append(entry)
    return demos

class BaseProcessor:
    # Class-level cache: once the main process scans the directory,
    # worker processes can reuse the result via _folder_cache.
    _folder_cache: dict = {}

    def __init__(self, cfg: DictConfig):
        # Store configuration for potential future use
        self.cfg = cfg

        # Apply configuration to instance attributes
        self._apply_config(cfg)

        # Validate configuration
        self._validate_config(cfg)

        # Load robot calibration file (if exists)
        self._load_calibration()

        # Set up paths and data folders
        self._setup_paths_and_folders(cfg)

        # Initialize camera parameters
        self._init_camera_parameters()

    def _apply_config(self, cfg: DictConfig) -> None:
        """Apply configuration to instance attributes."""
        # Basic attributes
        self.input_resolution = cfg.input_resolution
        self.output_resolution = cfg.output_resolution
        self.output_resize = cfg.output_resize
        self.project_folder = _get_project_folder()
        self.debug = cfg.debug
        self.n_processes = cfg.n_processes
        self.verbose = cfg.verbose
        self.skip_existing = cfg.skip_existing
        self.robot = cfg.robot
        self.gripper = cfg.gripper
        self.square = cfg.square
        self.epic = cfg.epic
        self.bimanual_setup = cfg.bimanual_setup
        self.target_hand = cfg.target_hand
        self.constrained_hand = cfg.constrained_hand
        self.depth_for_overlay = cfg.depth_for_overlay
        self.render = cfg.render
        self.debug_cameras = getattr(cfg, 'debug_cameras', [])

        # Apply bimanual setup logic
        if self.bimanual_setup != "single_arm":
            self.target_hand = "both"

    def _validate_config(self, cfg: DictConfig) -> None:
        """Validate critical configuration parameters."""
        if cfg.input_resolution <= 0 or cfg.output_resolution <= 0:
            raise ValueError(f"Resolutions must be positive: input={cfg.input_resolution}, output={cfg.output_resolution}")

        if not os.path.exists(cfg.data_root_dir):
            pass  # non-base processors read from processed_data_root_dir

        # camera_intrinsics might not exist for EgoDex/VITRA (generated during base stage)
        if not os.path.exists(cfg.camera_intrinsics):
            pass

    def _load_calibration(self) -> None:
        """Load robot-specific calibration from configs/calibration/{robot}.json (lowercase).

        Populates self.calibration with parsed numpy arrays, or None if no file found.
        """
        import json as _json
        calib_path = os.path.join(self.project_folder, "configs", "calibration",
                                  f"{self.robot.lower()}.json")
        if not os.path.exists(calib_path):
            self.calibration = None
            return

        with open(calib_path, "r") as f:
            raw = _json.load(f)

        self.calibration = {
            "head_T": np.array(raw["head_T"], dtype=np.float64),
            "left_right_T": np.array(raw["left_right_T"], dtype=np.float64),
            "left_armbase_offset": np.array(raw["left_armbase_offset"], dtype=np.float64),
            "intrinsics": raw["intrinsics"],
        }
        if "wrist_cam_T" in raw:
            self.calibration["wrist_cam_T"] = np.array(raw["wrist_cam_T"], dtype=np.float64)
        if "tcp_adjust" in raw:
            self.calibration["tcp_adjust"] = np.array(raw["tcp_adjust"], dtype=np.float64)
        logger.info("Loaded calibration for %s from %s", self.robot, calib_path)

    def _setup_paths_and_folders(self, cfg: DictConfig) -> None:
        """Set up paths configuration and create necessary directories."""
        self.data_folder = os.path.join(cfg.data_root_dir, cfg.demo_name)
        self.processed_data_folder = os.path.join(cfg.processed_data_root_dir, cfg.demo_name)

        # If data_folder doesn't exist, fall back to processed_data_folder
        # (non-base processors read from processed output, not raw data)
        if not os.path.exists(self.data_folder):
            self.data_folder = self.processed_data_folder

        os.makedirs(self.processed_data_folder, exist_ok=True)

        # Build a cache key from config values that affect folder listing
        cache_key = (
            self.data_folder,
            str(cfg.get("target_folders", None)),
            cfg.get("max_demos", None),
            cfg.get("demo_range", None),
            str(cfg.get("exclude_demos", None)),
        )

        # Reuse cached folder list if available (avoids repeated listdir on
        # directories with hundreds of thousands of entries).
        if cache_key in BaseProcessor._folder_cache:
            cached = BaseProcessor._folder_cache[cache_key]
            self.all_data_folders = cached
            return

        # Get all folders in data_folder
        try:
            # 1. List all subdirectories, sorted numerically
            all_data_folders = [d1 for d1 in os.listdir(self.data_folder)
                              if os.path.isdir(os.path.join(self.data_folder, d1))]
            # Filter to numeric-only folders (skip non-demo dirs like "tests/", "configs/")
            numeric_folders = [d for d in all_data_folders if d.isdigit()]
            sorted_folders = sorted(numeric_folders, key=lambda x: int(x))

            # 2. Restrict to an explicit list (target_folders)
            target_folders = cfg.get("target_folders", None)
            if target_folders is not None:
                if isinstance(target_folders, str):
                    target_list = [f.strip() for f in target_folders.split(',')]
                else:
                    target_list = [str(f) for f in target_folders]
                sorted_folders = [f for f in sorted_folders if f in target_list]
                if not sorted_folders:
                    print(f"Warning: none of {target_list} found in {self.data_folder}")

            # 3. Cap the count (max_demos)
            max_demos = cfg.get("max_demos", None)
            if max_demos is not None:
                sorted_folders = sorted_folders[:int(max_demos)]

            # 4. Restrict to an index range (demo_range)
            demo_range = cfg.get("demo_range", None)
            if demo_range is not None:
                range_str = str(demo_range).strip()
                if "-" in range_str:
                    parts = range_str.split("-", 1)
                    start = int(parts[0]) if parts[0] else 0
                    end = int(parts[1]) if parts[1] else len(sorted_folders)
                    sorted_folders = sorted_folders[start:end]
                    print(f"  demo_range={range_str} → index [{start}:{end}], {len(sorted_folders)} demos")

            # 5. Drop excluded demos (exclude_demos)
            exclude_demos = cfg.get("exclude_demos", None)
            if exclude_demos is not None:
                if isinstance(exclude_demos, str):
                    exclude_set = {f.strip() for f in exclude_demos.split(',')}
                else:
                    exclude_set = {str(f) for f in exclude_demos}
                before = len(sorted_folders)
                sorted_folders = [f for f in sorted_folders if f not in exclude_set]
                excluded = before - len(sorted_folders)
                if excluded:
                    print(f"  exclude_demos: removed {excluded} demos, {len(sorted_folders)} remaining")

            # 6. Store and cache
            self.all_data_folders = sorted_folders
            BaseProcessor._folder_cache[cache_key] = sorted_folders

            print(f"Folders to process ({len(self.all_data_folders)})")
        except OSError as e:
            if e.errno == errno.EACCES:
                raise PermissionError(f"Permission denied accessing data folder: {self.data_folder}")
            elif e.errno == errno.ENOENT:
                raise FileNotFoundError(f"Data folder not found: {self.data_folder}")
            else:
                raise RuntimeError(f"OS error accessing data folder {self.data_folder}: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid folder name format in {self.data_folder}. Folders should be numbered: {e}")

    def _init_camera_parameters(self) -> None:
        """Initialize camera intrinsics and extrinsics.

        When camera_intrinsics / camera_extrinsics are empty (--demo-dir mode),
        we set dummy values. Each processor loads real params from the demo folder.
        """
        intrinsics_path = self.cfg.camera_intrinsics
        if intrinsics_path and os.path.exists(intrinsics_path):
            self.intrinsics_dict, self.intrinsics_matrix = self.get_intrinsics(intrinsics_path)
        else:
            self.intrinsics_dict = {}
            self.intrinsics_matrix = np.eye(3)

        if hasattr(self.cfg, 'camera_extrinsics') and self.cfg.camera_extrinsics:
            camera_extrinsics_path = self.cfg.camera_extrinsics
        else:
            camera_extrinsics_path = self._get_camera_extrinsics_path()

        if os.path.exists(camera_extrinsics_path):
            self.T_cam2robot, self.extrinsics = self.get_extrinsics(camera_extrinsics_path)
        else:
            self.T_cam2robot = np.eye(4)
            self.extrinsics = {}

    def _get_camera_extrinsics_path(self) -> str:
        """Get the appropriate camera extrinsics path based on bimanual setup."""
        if self.bimanual_setup == "shoulders":
            return "camera/camera_extrinsics_ego_bimanual_shoulders.json"
        elif self.bimanual_setup == "single_arm":
            return "camera/camera_extrinsics.json"
        else:
            raise ValueError(f"Invalid bimanual setup: {self.bimanual_setup}. Must be 'single_arm' or 'shoulders'.")

    def get_paths(self, data_path: str) -> Paths:
        """
        Get all file paths for a demo.

        Args:
            data_path: Path to the demo data

        Returns:
            Paths object containing all file paths
        """
        paths = Paths(
            data_path=Path(data_path),
            robot_name=self.robot
        )
        paths.ensure_directories_exist()
        return paths

    def get_save_folder(self, data_sub_folder: str) -> str:
        data_sub_folder_fullpath = os.path.join(self.data_folder, str(data_sub_folder))
        save_folder = os.path.join(self.processed_data_folder, str(data_sub_folder))
        # Direct existence check — avoids scanning the entire parent directory
        if not os.path.isdir(save_folder):
            shutil.copytree(data_sub_folder_fullpath, save_folder)
        return save_folder

    def process_one_demo(self, data_sub_folder: str):
        """Base class stub — subclasses must override."""
        raise NotImplementedError

    @staticmethod
    def check_exists(data_sub_folder: str, cfg=None) -> bool:
        """Check if this processor's output already exists for a demo.

        Staticmethod so the pipeline can call it without instantiating the
        processor (which loads models). Override in subclasses.

        Args:
            data_sub_folder: For disk-based datasets this is the full demo path;
                for lerobot datasets it is just the episode index (e.g. "5"),
                and subclasses must use `cfg` to resolve the real save folder.
            cfg: optional config for resolving paths when data_sub_folder is a
                bare episode id.
        """
        return False

    def close(self):
        """Release resources (GPU memory, model handles, etc.)."""

    def get_intrinsics(self, intrinsics_path: str) -> Tuple[dict, np.ndarray]:
        intrinsics_matrix, intrinsics_dict = get_intrinsics_from_json(intrinsics_path)
        if self.square:
            intrinsics_dict, intrinsics_matrix = self.update_intrinsics_for_square_image(self.input_resolution,
                                                                                        intrinsics_dict,
                                                                                        intrinsics_matrix)
        return intrinsics_dict, intrinsics_matrix

    def get_extrinsics(self, extrinsics_path: str) -> Tuple[np.ndarray, dict]:
        """Load and process camera extrinsics from JSON file.

        Args:
            extrinsics_path: Path to the extrinsics JSON file

        Returns:
            Tuple of (transformation_matrix, extrinsics_dict)

        Raises:
            FileNotFoundError: If extrinsics file doesn't exist
            json.JSONDecodeError: If extrinsics file is invalid JSON
            ValueError: If extrinsics data is invalid
        """
        if not os.path.exists(extrinsics_path):
            raise FileNotFoundError(f"Camera extrinsics file not found: {extrinsics_path}")

        try:
            with open(extrinsics_path, "r") as f:
                camera_extrinsics = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in extrinsics file {extrinsics_path}: {str(e)}")

        try:
            T_cam2robot = get_transformation_matrix_from_extrinsics(camera_extrinsics)
        except Exception as e:
            raise ValueError(f"Failed to process extrinsics data from {extrinsics_path}: {str(e)}")

        return T_cam2robot, camera_extrinsics

    # ====================== Per-frame extrinsics ======================
    def _load_per_frame_extrinsics(self, paths) -> Optional[dict]:
        """
        Load per-frame camera extrinsics from cam_extrinsics.npz. Returns None
        when the file is absent, in which case callers fall back to a fixed
        T_cam2robot.

        Returns:
            dict with keys: R_c2w (T,3,3), t_c2w (T,3), or None
        """
        extrinsics_path = paths.cam_extrinsics
        if extrinsics_path is not None and os.path.exists(str(extrinsics_path)):
            data = np.load(str(extrinsics_path))
            result = {
                'R_c2w': data['R_c2w'],
                't_c2w': data['t_c2w'],
                'R_w2c': data['R_w2c'],
                't_w2c': data['t_w2c'],
            }
            return result
        else:
            logger.info("No per-frame extrinsics found, will use fixed T_cam2robot")
            return None

    def _load_camera_intrinsics(self, paths) -> Optional[np.ndarray]:
        """
        Load the camera intrinsics matrix K (3x3) from cam_intrinsics.json.

        Returns:
            K: (3, 3) numpy array, or None
        """
        import json
        intrinsics_path = str(paths.cam_intrinsics)
        if not os.path.exists(intrinsics_path):
            return None
        with open(intrinsics_path, 'r') as f:
            data = json.load(f)
        # Accepts either {"left": {"fx":..}} or {"fx":..}
        if "left" in data:
            params = data["left"]
        else:
            params = data
        fx = params["fx"]
        fy = params["fy"]
        cx = params["cx"]
        cy = params["cy"]
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)

        intrinsics_dict = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "new_w": params["new_w"],
            "new_h": params["new_h"],
            "orig_w": params["orig_w"],
            "orig_h": params["orig_h"],
            "v_fov": params["v_fov"],
            "fps": params["fps"],
        }

        return K, intrinsics_dict
    # ====================================================================

    def load_sampled_depths(self, depth_dir: str, indices: npt.NDArray[np.int_] | List[int]) -> np.ndarray:
        """
        Read selected depth frames from a directory of per-frame files.

        Equivalent to np.load(paths.depth)[indices] but avoids loading the whole
        stack into memory.

        Args:
            depth_dir: directory holding per-frame depth files
            indices: frame indices to read

        Returns:
            np.ndarray of shape (len(indices), H, W)
        """
        # Natural sort is required so file order matches frame order
        depth_files = natsorted(glob(os.path.join(depth_dir, "*.npz")))
        use_npz = True
        if not depth_files:
            depth_files = natsorted(glob(os.path.join(depth_dir, "*.npy")))
            use_npz = False

        selected_depths = []
        for idx in indices:
            if 0 <= idx < len(depth_files):
                file_path = depth_files[idx]
                if use_npz:
                    depth_data = np.load(file_path)['depth']
                else:
                    depth_data = np.load(file_path)
                selected_depths.append(depth_data)
            else:
                raise IndexError(f"Index {idx} out of bounds for depth files (total {len(depth_files)})")

        # Stack to (N, H, W)
        return np.stack(selected_depths, axis=0)

    @staticmethod
    def update_intrinsics_for_square_image(img_h: int, intrinsics_dict: dict,
                                           intrinsics_matrix: np.ndarray) -> Tuple[dict, np.ndarray]:
        """
        Adjusts camera intrinsic parameters for a square image by modifying the principal point offset.

        Args:
            img_h (int): Height of the image (assumed to be square).
            intrinsics_dict (dict): Dictionary of intrinsic parameters.
            intrinsics_matrix (np.ndarray): Intrinsic matrix.

        Returns:
            Tuple[dict, np.ndarray]: Updated intrinsic parameters and matrix.
        """
        img_w = img_h * 16 // 9
        offset = (img_w - img_h) // 2
        intrinsics_dict["cx"] -= offset
        intrinsics_matrix[0, 2] -= offset
        return intrinsics_dict, intrinsics_matrix

    def draw_eef_on_image(
        self,
        img: np.ndarray,
        ee_pt_world: np.ndarray,
        ee_ori_world: np.ndarray,
        ee_width: float,
        R_w2c: np.ndarray,
        t_w2c: np.ndarray,
        K: np.ndarray,
        color: Tuple[int, int, int],
        hand_side: str,
        line_thickness: int = 2,
        circle_radius: int = 4,
        axis_length: float = 0.08,
        is_detected: Optional[bool] = None,
        text_y_offset: Optional[int] = None,
        short_axis_threshold: float = 12.0,
        jaw_pts_world: Optional[np.ndarray] = None,
    ) -> None:
        """
        Draw a single end-effector pose on an image (modified in-place).

        Args:
            img:             BGR image, modified in-place.
            ee_pt_world:     EEF position in world frame (3,).
            ee_ori_world:    EEF rotation matrix in world frame (3,3).
            ee_width:        Gripper opening width in meters.
            R_w2c:           World→camera rotation (3,3).
            t_w2c:           World→camera translation (3,).
            K:               Camera intrinsics (3,3).
            color:           BGR colour for this hand.
            hand_side:       "left" or "right".
            line_thickness:  Axis line thickness.
            circle_radius:   Center dot radius.
            axis_length:     Axis length in meters.
            is_detected:     True=detected, False=carry-forward, None=don't distinguish.
            text_y_offset:   Text vertical offset in pixels.
            short_axis_threshold: Below this pixel length, use depth symbol instead of line.
        """
        # --- Project center ---
        ee_pt_cam = R_w2c @ ee_pt_world + t_w2c
        if ee_pt_cam[2] <= 0.01:
            return

        uv = K @ ee_pt_cam
        uc = int(uv[0] / uv[2])
        vc = int(uv[1] / uv[2])

        # --- Layer 1: Gripper jaw line ---
        light_color = tuple(min(255, c + 80) for c in color)
        if jaw_pts_world is not None and len(jaw_pts_world) == 2:
            pts_2d = []
            for pt in jaw_pts_world:
                jaw_c = R_w2c @ pt + t_w2c
                if jaw_c[2] > 0.01:
                    uv_j = K @ jaw_c
                    pts_2d.append((int(uv_j[0] / uv_j[2]), int(uv_j[1] / uv_j[2])))
            if len(pts_2d) == 2:
                cv2.line(img, pts_2d[0], pts_2d[1], light_color, 1, cv2.LINE_AA)
                cv2.circle(img, pts_2d[0], 2, light_color, -1)
                cv2.circle(img, pts_2d[1], 2, light_color, -1)
        else:
            half_w = ee_width / 2.0
            for sign in [1, -1]:
                jaw_w = ee_pt_world + sign * ee_ori_world[:, 2] * half_w
                jaw_c = R_w2c @ jaw_w + t_w2c
                if jaw_c[2] > 0.01:
                    uv_j = K @ jaw_c
                    uj = int(uv_j[0] / uv_j[2])
                    vj = int(uv_j[1] / uv_j[2])
                    cv2.line(img, (uc, vc), (uj, vj), light_color, 1, cv2.LINE_AA)
                    cv2.circle(img, (uj, vj), 2, light_color, -1)

        # --- Layer 2: Axes (sorted: longest first → shortest last = on top) ---
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        axis_labels = ["X", "Y", "Z"]

        axis_info = []
        for ax_i in range(3):
            end_world = ee_pt_world + ee_ori_world[:, ax_i] * axis_length
            end_cam = R_w2c @ end_world + t_w2c
            if end_cam[2] <= 0.01:
                axis_info.append((ax_i, None, None, None, 0.0))
                continue
            uv_e = K @ end_cam
            ue = int(uv_e[0] / uv_e[2])
            ve = int(uv_e[1] / uv_e[2])
            proj_len = np.sqrt((ue - uc) ** 2 + (ve - vc) ** 2)
            going_away = end_cam[2] > ee_pt_cam[2]
            axis_info.append((ax_i, ue, ve, going_away, proj_len))

        # Draw long axes first so short axes end up on top
        axis_info.sort(key=lambda x: -x[4])

        for ax_i, ue, ve, going_away, proj_len in axis_info:
            if ue is None:
                continue
            ac = axis_colors[ax_i]
            label = axis_labels[ax_i]

            if proj_len < short_axis_threshold:
                # ---- Axis nearly perpendicular to the image plane: draw a symbol ----
                # Offset to clear the centre point
                ox = uc + (circle_radius + 10) * (1 if ax_i != 2 else -1)
                oy = vc + (circle_radius + 10) * (-1 if ax_i == 0 else 1)

                if going_away:
                    # Pointing away from the camera
                    r = 6
                    cv2.circle(img, (ox, oy), r, ac, 2, cv2.LINE_AA)
                    cv2.line(img, (ox - r + 1, oy - r + 1), (ox + r - 1, oy + r - 1), ac, 1)
                    cv2.line(img, (ox - r + 1, oy + r - 1), (ox + r - 1, oy - r + 1), ac, 1)
                    cv2.putText(img, label, (ox + r + 2, oy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, ac, 1)
                else:
                    # Pointing towards the camera
                    r = 6
                    cv2.circle(img, (ox, oy), r, ac, 2, cv2.LINE_AA)
                    cv2.circle(img, (ox, oy), 2, ac, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (ox + r + 2, oy + 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, ac, 1)

                # Thin dashed line back to the centre to show which EEF it belongs to
                self._draw_thin_dash(img, (uc, vc), (ox, oy), ac)
            else:
                # ---- Normal length: draw an arrow ----
                if going_away:
                    cv2.arrowedLine(img, (uc, vc), (ue, ve), ac,
                                    line_thickness, cv2.LINE_AA, tipLength=0.2)
                else:
                    self._draw_dashed_arrow(img, (uc, vc), (ue, ve), ac,
                                    line_thickness, dash_length=10, gap_length=6)
                cv2.putText(img, label, (ue + 5, ve - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, ac, 1)

        # --- Layer 3: Center dot (smallest, on top) ---
        mk_color = color
        if is_detected is not None and not is_detected:
            mk_color = (128, 128, 128)
        cv2.circle(img, (uc, vc), circle_radius, mk_color, -1)
        cv2.circle(img, (uc, vc), circle_radius + 1, (255, 255, 255), 1)

        # --- Text ---
        ty = text_y_offset if text_y_offset is not None else (25 if hand_side == "left" else 75)
        det_str = ""
        if is_detected is not None:
            det_str = " DET" if is_detected else " CFW"
        cv2.putText(img, f"{hand_side[0].upper()}:{det_str} w={ee_width:.3f}m",
                    (10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mk_color, 1)
        cv2.putText(img,
                    f"  ({ee_pt_world[0]:.3f},{ee_pt_world[1]:.3f},{ee_pt_world[2]:.3f})",
                    (10, ty + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, mk_color, 1)


    def _draw_thin_dash(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        dash_len: int = 3,
        gap_len: int = 4,
    ) -> None:
        """Draw a thin dashed line connecting the symbol back to the center."""
        x1, y1 = pt1
        x2, y2 = pt2
        total = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if total < 1:
            return
        ux, uy = (x2 - x1) / total, (y2 - y1) / total
        d = 0.0
        while d < total:
            s = d
            e = min(d + dash_len, total)
            cv2.line(img,
                    (int(x1 + ux * s), int(y1 + uy * s)),
                    (int(x1 + ux * e), int(y1 + uy * e)),
                    color, 1, cv2.LINE_AA)
            d += dash_len + gap_len


    def _draw_dashed_arrow(
        self,
        img: np.ndarray,
        pt1: Tuple[int, int],
        pt2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10,
        gap_length: int = 6,
    ) -> None:
        """Dashed line with hollow arrowhead (axis pointing toward camera)."""
        x1, y1 = pt1
        x2, y2 = pt2
        total = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if total < 1:
            return
        ux, uy = (x2 - x1) / total, (y2 - y1) / total

        d = 0.0
        while d < total:
            s = d
            e = min(d + dash_length, total)
            cv2.line(img,
                    (int(x1 + ux * s), int(y1 + uy * s)),
                    (int(x1 + ux * e), int(y1 + uy * e)),
                    color, thickness, cv2.LINE_AA)
            d += dash_length + gap_length

        # Hollow arrowhead
        arrow_len = 14
        arrow_hw = 6
        bx = x2 - ux * arrow_len
        by = y2 - uy * arrow_len
        px, py = -uy, ux
        pts = np.array([
            [int(x2), int(y2)],
            [int(bx + px * arrow_hw), int(by + py * arrow_hw)],
            [int(bx - px * arrow_hw), int(by - py * arrow_hw)],
        ], dtype=np.int32)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)

    def draw_eef_on_image_camframe(
        self,
        img: np.ndarray,
        ee_pt_cam: np.ndarray,
        ee_ori_cam: np.ndarray,
        K: np.ndarray,
        color: Tuple[int, int, int],
        hand_side: str,
        ee_width: float = 0.0,
        line_thickness: int = 2,
        circle_radius: int = 4,
        axis_length: float = 0.05,
        is_detected: Optional[bool] = None,
        text_y_offset: Optional[int] = None,
        short_axis_threshold: float = 12.0,
    ) -> None:
        """
        Convenience wrapper for data already in camera frame.

        Equivalent to draw_eef_on_image with R_w2c=I, t_w2c=0.

        Args:
            ee_pt_cam:   EEF position in camera frame (3,).
            ee_ori_cam:  EEF rotation matrix in camera frame (3,3).
            K:           Camera intrinsics (3,3).
            ee_width:    Gripper width in meters (0 = don't draw jaw line).
            Others:      Same as draw_eef_on_image.
        """
        R_identity = np.eye(3, dtype=np.float64)
        t_zero = np.zeros(3, dtype=np.float64)

        self.draw_eef_on_image(
            img=img,
            ee_pt_world=ee_pt_cam,      # "world" = camera frame
            ee_ori_world=ee_ori_cam,
            ee_width=ee_width,
            R_w2c=R_identity,
            t_w2c=t_zero,
            K=K,
            color=color,
            hand_side=hand_side,
            line_thickness=line_thickness,
            circle_radius=circle_radius,
            axis_length=axis_length,
            is_detected=is_detected,
            text_y_offset=text_y_offset,
            short_axis_threshold=short_axis_threshold,
        )

    def visualize_eef_overlay(
        self,
        video_path: str,
        output_path: str,
        K: np.ndarray,
        extrinsics: dict,
        union_indices: np.ndarray,
        data_groups: List[Dict[str, dict]],
        group_labels: Optional[List[str]] = None,
        sequences: Optional[Dict[str, object]] = None,
    ) -> None:
        """
        Generic EEF-overlay visualisation that works for both ActionProcessor and
        SmoothingProcessor.

        The function renders *len(data_groups)* side-by-side panels.  Each panel
        shows the same source video frame with a different set of EEF trajectories
        overlaid.  This covers two common use-cases:

        * **Single panel** (ActionProcessor): pass one data group.
        * **Two panels – raw vs smoothed** (SmoothingProcessor): pass two groups.

        Args:
            video_path:    Path to the source video file.
            output_path:   Path where the output mp4 will be written.
            K:             Camera intrinsic matrix, shape (3,3).
            extrinsics:    Dict with keys ``R_w2c`` (T,3,3) and ``t_w2c`` (T,3).
            union_indices: 1-D int array mapping *action index* → *frame index*.
            data_groups:   List of groups, where each group is::

                            {
                                "left":  {"ee_pts": (A,3), "ee_oris": (A,3,3), "ee_widths": (A,)},
                                "right": {"ee_pts": ..., ...},
                            }

                        A group may contain only "left", only "right", or both.
                        A = len(union_indices).
            group_labels:  Optional display label for each group
                        (e.g. ``["RAW", "SMOOTHED"]``).  Defaults to
                        ``["Panel 0", "Panel 1", ...]``.
            sequences:     Optional dict ``{"left": HandSequence, "right": HandSequence}``.
                        When provided, the ``hand_detected`` flag is forwarded to
                        ``draw_eef_on_image`` so that carry-forward frames are
                        rendered in gray.

        Visual parameters that differ between single-panel / comparison modes are
        chosen automatically:

        +-----------+------------------+----------------+
        | panels    | line_thickness   | circle_radius  |
        +===========+==================+================+
        | 1         | 2                | 8              |
        +-----------+------------------+----------------+
        | ≥ 2       | 2 (first) / 3   | 6 / 8          |
        +-----------+------------------+----------------+
        """
        if not os.path.exists(video_path):
            logger.warning("Video not found: %s", video_path)
            return

        R_w2c = extrinsics["R_w2c"]
        t_w2c = extrinsics["t_w2c"]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video: %s", video_path)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        n_panels = len(data_groups)
        if group_labels is None:
            group_labels = [f"Panel {i}" for i in range(n_panels)]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (W * n_panels, H))

        union_to_idx = {int(fi): ai for ai, fi in enumerate(union_indices)}

        # Default style: smaller circle, longer axes
        default_styles: List[dict] = []
        for gi in range(n_panels):
            if n_panels == 1:
                default_styles.append({
                    "line_thickness": 2,
                    "circle_radius": 4,
                    "axis_length": 0.08,
                })
            else:
                if gi == 0:  # RAW panel
                    default_styles.append({
                        "line_thickness": 2,
                        "circle_radius": 3,
                        "axis_length": 0.08,
                    })
                else:  # SMOOTHED panel
                    default_styles.append({
                        "line_thickness": 3,
                        "circle_radius": 4,
                        "axis_length": 0.08,
                    })

        T_ext = R_w2c.shape[0]
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= T_ext:
                break

            panels: List[np.ndarray] = []
            R_w2c_t = R_w2c[frame_idx]
            t_w2c_t = t_w2c[frame_idx]

            for gi, group in enumerate(data_groups):
                img = frame.copy()

                if frame_idx in union_to_idx:
                    action_idx = union_to_idx[frame_idx]

                    for hand_side in ("left", "right"):
                        if hand_side not in group:
                            continue
                        hand_data = group[hand_side]

                        if action_idx >= len(hand_data["ee_pts"]):
                            continue

                        ee_pt = hand_data["ee_pts"][action_idx]
                        ee_ori = hand_data["ee_oris"][action_idx]
                        ee_w = hand_data["ee_widths"][action_idx]

                        # Determine detection flag if sequences are available
                        is_det: Optional[bool] = None
                        if sequences is not None and hand_side in sequences:
                            hs = sequences[hand_side]
                            if frame_idx < len(hs.hand_detected):
                                is_det = bool(hs.hand_detected[frame_idx])

                        jaw_kw = {}
                        jaw = hand_data.get("jaw_pts")
                        if jaw is not None and action_idx < len(jaw):
                            jaw_kw["jaw_pts_world"] = jaw[action_idx]

                        self.draw_eef_on_image(
                            img,
                            ee_pt,
                            ee_ori,
                            ee_w,
                            R_w2c_t,
                            t_w2c_t,
                            K,
                            color=_HAND_COLORS.get(hand_side, (200, 200, 200)),
                            hand_side=hand_side,
                            is_detected=is_det,
                            **default_styles[gi],
                            **jaw_kw,
                        )

                # Panel label
                label_color = (255, 255, 255) if gi == 0 else (0, 255, 255)
                cv2.putText(
                    img, f"{group_labels[gi]}  F:{frame_idx}",
                    (10, H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1,
                )
                panels.append(img)

            combined = np.hstack(panels) if n_panels > 1 else panels[0]
            out_writer.write(combined)
            frame_idx += 1

        cap.release()
        out_writer.release()
        self.ensure_h264(output_path)

    @staticmethod
    def ensure_h264(video_path: str) -> None:
        """
        Re-encode a video to H.264/MP4 so it plays in VS Code / browsers.

        Replaces the file in-place. Skips if ffmpeg is not available or if
        the file is already H.264.
        """
        import shutil
        import subprocess
        video_path = str(video_path)
        if not os.path.exists(video_path):
            return
        tmp = video_path + ".h264.mp4"
        cmd_args = ["-y", "-i", video_path,
                    "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-an", tmp]
        ffmpeg_candidates = [shutil.which("ffmpeg") or "ffmpeg", "/usr/bin/ffmpeg"]
        for ffmpeg_bin in ffmpeg_candidates:
            try:
                subprocess.run([ffmpeg_bin] + cmd_args,
                               check=True, capture_output=True)
                os.replace(tmp, video_path)
                return
            except (FileNotFoundError, subprocess.CalledProcessError):
                if os.path.exists(tmp):
                    os.remove(tmp)
