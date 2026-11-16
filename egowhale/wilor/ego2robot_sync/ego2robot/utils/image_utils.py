import json
import numpy as np
import os
import mediapy as media
from typing import Dict, List, Tuple


def get_transformation_matrix_from_extrinsics(camera_extrinsics: List[Dict]) -> np.ndarray:
    """Get homogeneous transformation matrix from camera extrinsics."""
    cam_base_pos = np.array(camera_extrinsics[0]["camera_base_pos"])
    cam_base_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, 3] = cam_base_pos
    T_cam2robot[:3, :3] = np.array(cam_base_ori).reshape(3, 3)
    return T_cam2robot


def get_intrinsics_from_json(json_path: str) -> Tuple[np.ndarray, dict]:
    with open(json_path, "r") as f:
        camera_intrinsics = json.load(f)


    # Get camera matrix
    fx = camera_intrinsics["left"]["fx"]
    fy = camera_intrinsics["left"]["fy"]
    cx = camera_intrinsics["left"]["cx"]
    cy = camera_intrinsics["left"]["cy"]
    v_fov = camera_intrinsics["left"]["v_fov"]
    intrinsics_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    intrinsics_dict = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        # "w": camera_intrinsics["left"]["w"],
        # "h": camera_intrinsics["left"]["h"],
        "v_fov": v_fov,
    }

    return intrinsics_matrix, intrinsics_dict
