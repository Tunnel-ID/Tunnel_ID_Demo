from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# Coarse canonical 3D points aligned with MediaPipe face detection keypoint semantics.
MODEL_3D = np.array(
    [
        [-30.0, -30.0, -30.0],  # right eye
        [30.0, -30.0, -30.0],  # left eye
        [0.0, 0.0, 0.0],  # nose
        [0.0, 30.0, -20.0],  # mouth center
        [-65.0, -5.0, -15.0],  # right ear tragion
        [65.0, -5.0, -15.0],  # left ear tragion
    ],
    dtype=np.float64,
)


def _rotation_matrix_to_euler_angles(r_mat: np.ndarray) -> Tuple[float, float, float]:
    sy = float(np.sqrt(r_mat[0, 0] * r_mat[0, 0] + r_mat[1, 0] * r_mat[1, 0]))
    singular = sy < 1e-6

    if not singular:
        x = float(np.arctan2(r_mat[2, 1], r_mat[2, 2]))
        y = float(np.arctan2(-r_mat[2, 0], sy))
        z = float(np.arctan2(r_mat[1, 0], r_mat[0, 0]))
    else:
        x = float(np.arctan2(-r_mat[1, 2], r_mat[1, 1]))
        y = float(np.arctan2(-r_mat[2, 0], sy))
        z = 0.0

    return np.degrees(y), np.degrees(x), np.degrees(z)


def estimate_head_pose(
    image_shape: Tuple[int, int, int],
    keypoints_xy: np.ndarray,
) -> Tuple[float, float, float]:
    h, w = image_shape[:2]
    if keypoints_xy.shape[0] < 6:
        return 0.0, 0.0, 0.0

    focal = float(w)
    cam = np.array([[focal, 0, w / 2.0], [0, focal, h / 2.0], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    image_points = keypoints_xy[:6].astype(np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0

    r_mat, _ = cv2.Rodrigues(rvec)
    yaw_deg, pitch_deg, roll_deg = _rotation_matrix_to_euler_angles(r_mat)
    return float(yaw_deg), float(pitch_deg), float(roll_deg)
