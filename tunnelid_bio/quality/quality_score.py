from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from tunnelid_bio.types import QualityMetrics


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def quality_score(
    metrics: Iterable[QualityMetrics],
    liveness_confidence: float,
    intra_session_variance: float,
) -> float:
    rows = list(metrics)
    if not rows:
        return 0.0

    sharpness = np.mean([m.sharpness for m in rows])
    exposure = np.mean([m.exposure for m in rows])
    occlusion = np.mean([m.occlusion_ratio for m in rows])
    blur = np.mean([m.motion_blur for m in rows])
    pose_error = np.mean([m.pose_error for m in rows])

    sharp_s = _clamp01(sharpness / 250.0)
    exposure_s = _clamp01(1.0 - abs(exposure - 127.5) / 127.5)
    occlusion_s = _clamp01(1.0 - occlusion)
    blur_s = _clamp01(1.0 - min(1.0, blur / 300.0))
    pose_s = _clamp01(1.0 - min(1.0, pose_error / 15.0))
    variance_s = _clamp01(1.0 - min(1.0, intra_session_variance / 0.05))
    liveness_s = _clamp01(liveness_confidence)

    # Weighted quality score aligned with capture reliability for matching + FE tolerance.
    score = (
        0.20 * sharp_s
        + 0.14 * exposure_s
        + 0.16 * occlusion_s
        + 0.14 * blur_s
        + 0.16 * pose_s
        + 0.12 * variance_s
        + 0.08 * liveness_s
    )
    return _clamp01(score)


def quality_gate(score: float, threshold: float) -> Tuple[bool, str]:
    passed = score >= threshold
    return passed, ("OK" if passed else "REJECT_LOW_QUALITY_SCORE")


def summarize_quality(metrics: Iterable[QualityMetrics]) -> Dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {
            "mean_sharpness": 0.0,
            "mean_exposure": 0.0,
            "mean_occlusion_ratio": 1.0,
            "mean_motion_blur": 999.0,
            "mean_pose_error": 999.0,
        }

    return {
        "mean_sharpness": float(np.mean([m.sharpness for m in rows])),
        "mean_exposure": float(np.mean([m.exposure for m in rows])),
        "mean_occlusion_ratio": float(np.mean([m.occlusion_ratio for m in rows])),
        "mean_motion_blur": float(np.mean([m.motion_blur for m in rows])),
        "mean_pose_error": float(np.mean([m.pose_error for m in rows])),
    }
