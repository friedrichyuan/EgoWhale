"""Path management for pipeline outputs."""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    """Data class containing all file paths used by processors."""
    data_path: Path
    robot_name: str = "franka"

    def __post_init__(self):
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)

        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {self.data_path}")

        # Videos
        self.video_left = self.data_path / "video_L.mp4"
        self.video_rgb_imgs = self.data_path / "video_rgb_imgs.mkv"

        # Intrinsics / extrinsics
        self.cam_intrinsics = self.data_path / "cam_intrinsics.json"
        self.cam_extrinsics = self.data_path / "cam_extrinsics.npz"
        self.info = self.data_path / "info.json"

        # Depth
        self.depth = self.data_path / "depth.npy"
        self.depth_path = self.data_path / "depth"

        # Bbox (optional pre-computed bounding boxes)
        self.video_bboxes = self.data_path / "bbox_processor" / "video_bboxes.mkv"

        # Segmentation processor
        self.segmentation_processor = self.data_path / "segmentation_processor"
        self.masks_arm = self.segmentation_processor / "masks_arm.npy"
        self.video_masks_arm = self.segmentation_processor / "video_masks_arm.mkv"
        self.video_sam_arm = self.segmentation_processor / "video_sam_arm.mkv"
        for side in ["left", "right"]:
            setattr(self, f"masks_hand_{side}", self.segmentation_processor / f"masks_hand_{side}.npy")
            setattr(self, f"video_masks_hand_{side}", self.segmentation_processor / f"video_masks_hand_{side}.mkv")
            setattr(self, f"video_sam_hand_{side}", self.segmentation_processor / f"video_sam_hand_{side}.mkv")

        # Hand processor
        self.hand_processor = self.data_path / "hand_processor"
        for side in ["left", "right"]:
            setattr(self, f"hand_data_{side}", self.hand_processor / f"hand_data_{side}.npz")
            setattr(self, f"hand_data_3d_{side}", self.hand_processor / f"hand_data_3d_{side}.npz")
        self.video_annot = self.data_path / "video_annot.mp4"

        # Action processor
        self.action_processor = self.data_path / "action_processor"
        for side in ["left", "right"]:
            setattr(self, f"actions_{side}", self.action_processor / f"actions_{side}.npz")

        # Smoothing processor
        self.smoothing_processor = self.data_path / "smoothing_processor"
        for side in ["left", "right"]:
            setattr(self, f"smoothed_actions_{side}", self.smoothing_processor / f"smoothed_actions_{side}.npz")

        # Inpaint processor
        self.inpaint_processor = self.data_path / "inpaint_processor"
        self.video_overlay = self.data_path / "video_overlay.mkv"
        self.video_human_inpaint = self.inpaint_processor / "video_human_inpaint.mkv"
        self.training_data = self.data_path / "training_data.npz"

    def ensure_directories_exist(self):
        """Create data_path if it doesn't exist."""
        if isinstance(self.data_path, Path) and not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)
