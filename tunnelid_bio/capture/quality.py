from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from tunnelid_bio.config import PipelineConfig
from tunnelid_bio.types import FaceObservation, QualityMetrics


@dataclass(frozen=True)
class QualityScores:
    sharpness: float
    exposure: float
    occlusion_ratio: float
    motion_blur: float


def _crop(image: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox_xyxy
    h, w = image.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    return image[y1:y2, x1:x2]


def compute_quality_scores(image_bgr: np.ndarray, face: FaceObservation) -> QualityScores:
    roi = _crop(image_bgr, face.bbox_xyxy)
    if roi.size == 0:
        return QualityScores(0.0, 0.0, 1.0, 999.0)

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exposure = float(np.mean(gray))

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    skin_mask = cv2.inRange(hsv, (0, 15, 40), (35, 200, 255))
    skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size + 1e-12)
    occlusion_ratio = float(np.clip(1.0 - skin_ratio, 0.0, 1.0))

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_energy = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))
    motion_blur = float(1000.0 / (edge_energy + 1e-6))

    return QualityScores(
        sharpness=sharpness,
        exposure=exposure,
        occlusion_ratio=occlusion_ratio,
        motion_blur=motion_blur,
    )


def pose_error_for_target(face: FaceObservation, pose_name: str, cfg: PipelineConfig) -> float:
    bands = cfg.pose_bands
    target_map = {
        "center": (bands.center_yaw, bands.center_pitch),
        "yaw_left": (bands.yaw_left, bands.center_pitch),
        "yaw_right": (bands.yaw_right, bands.center_pitch),
        "pitch_up": (bands.center_yaw, bands.pitch_up),
        "pitch_down": (bands.center_yaw, bands.pitch_down),
    }
    target_yaw, target_pitch = target_map[pose_name]
    return float(np.sqrt((face.yaw_deg - target_yaw) ** 2 + (face.pitch_deg - target_pitch) ** 2))


def quality_gate(
    image_bgr: np.ndarray,
    face: FaceObservation,
    pose_name: str,
    cfg: PipelineConfig,
) -> QualityMetrics:
    score = compute_quality_scores(image_bgr, face)
    pose_error = pose_error_for_target(face, pose_name, cfg)

    qcfg = cfg.quality
    passes = (
        face.detection_confidence >= cfg.capture.min_detection_confidence
        and score.sharpness >= qcfg.min_sharpness
        and qcfg.min_exposure <= score.exposure <= qcfg.max_exposure
        and score.occlusion_ratio <= qcfg.max_occlusion_ratio
        and score.motion_blur <= qcfg.max_motion_blur
        and pose_error <= cfg.capture.pose_tolerance_deg
    )

    return QualityMetrics(
        sharpness=score.sharpness,
        exposure=score.exposure,
        occlusion_ratio=score.occlusion_ratio,
        motion_blur=score.motion_blur,
        pose_error=pose_error,
        passes=passes,
    )


def quality_weight(metrics: QualityMetrics) -> float:
    sharp_w = min(1.0, metrics.sharpness / 250.0)
    exp_mid = 127.5
    exp_w = max(0.0, 1.0 - (abs(metrics.exposure - exp_mid) / exp_mid))
    occ_w = max(0.0, 1.0 - metrics.occlusion_ratio)
    blur_w = max(0.0, 1.0 - min(1.0, metrics.motion_blur / 300.0))
    pose_w = max(0.0, 1.0 - min(1.0, metrics.pose_error / 15.0))
    return float(0.30 * sharp_w + 0.20 * exp_w + 0.20 * occ_w + 0.15 * blur_w + 0.15 * pose_w)
