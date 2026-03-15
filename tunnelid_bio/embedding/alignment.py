from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# Reference points for 112x112 aligned crops (ArcFace-like convention).
REFERENCE_5PTS = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _build_five_points_from_six(kp6_xy: np.ndarray) -> np.ndarray:
    right_eye = kp6_xy[0]
    left_eye = kp6_xy[1]
    nose = kp6_xy[2]
    mouth_center = kp6_xy[3]
    # Approximate mouth corners from mouth center and eye baseline for stable affine fit.
    eye_dist = np.linalg.norm(left_eye - right_eye)
    offset = np.array([0.18 * eye_dist, 0.0], dtype=np.float32)
    mouth_left = mouth_center - offset
    mouth_right = mouth_center + offset
    return np.stack([right_eye, left_eye, nose, mouth_left, mouth_right], axis=0).astype(np.float32)


def align_face_112(image_bgr: np.ndarray, kp6_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    src = _build_five_points_from_six(kp6_xy)
    dst = REFERENCE_5PTS
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        matrix = cv2.getAffineTransform(src[:3], dst[:3])
    aligned = cv2.warpAffine(image_bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR)
    return aligned, matrix
