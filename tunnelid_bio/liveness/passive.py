from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from tunnelid_bio.config import PassiveLivenessConfig
from tunnelid_bio.types import FaceObservation, FramePacket


@dataclass(frozen=True)
class PassiveScores:
    texture_score: float
    lbp_score: float
    moire_score: float
    banding_score: float
    frequency_score: float
    specular_dynamic_score: float


def _crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(1, min(w, x2))
    y2 = max(1, min(h, y2))
    return frame[y1:y2, x1:x2]


def _lbp_score(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    c = g[1:-1, 1:-1]
    code = np.zeros_like(c, dtype=np.uint8)
    neighbors = [
        g[:-2, :-2],
        g[:-2, 1:-1],
        g[:-2, 2:],
        g[1:-1, 2:],
        g[2:, 2:],
        g[2:, 1:-1],
        g[2:, :-2],
        g[1:-1, :-2],
    ]
    for i, n in enumerate(neighbors):
        code |= ((n >= c).astype(np.uint8) << i)

    hist = np.bincount(code.ravel(), minlength=256).astype(np.float64)
    hist /= (hist.sum() + 1e-12)
    entropy = float(-np.sum(hist * np.log(hist + 1e-12)))
    return float(np.clip(entropy / 5.4, 0.0, 1.0))


def _moire_score(gray128: np.ndarray) -> float:
    freq = np.fft.fftshift(np.fft.fft2(gray128.astype(np.float32)))
    mag = np.abs(freq)
    mag /= float(np.mean(mag) + 1e-6)

    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring = (rr > 12) & (rr < 48)
    vals = mag[ring]

    if vals.size == 0:
        return 0.0

    p995 = float(np.quantile(vals, 0.995))
    p95 = float(np.quantile(vals, 0.95) + 1e-6)
    peakiness = p995 / p95
    return float(np.clip((peakiness - 1.0) / 2.5, 0.0, 1.0))


def _banding_score(gray128: np.ndarray) -> float:
    row = np.mean(gray128.astype(np.float32), axis=1)
    row = row - cv2.GaussianBlur(row.reshape(-1, 1), (1, 9), 0).reshape(-1)
    spec = np.abs(np.fft.rfft(row))
    if spec.size < 4:
        return 0.0
    spec[0] = 0.0
    peak = float(np.max(spec))
    base = float(np.mean(spec) + 1e-6)
    return float(np.clip((peak / base - 1.0) / 8.0, 0.0, 1.0))


def _frequency_score(gray128: np.ndarray) -> float:
    freq = np.fft.fftshift(np.fft.fft2(gray128.astype(np.float32)))
    mag = np.abs(freq)
    ps = (mag ** 2).ravel()
    ps /= float(ps.sum() + 1e-12)
    entropy = float(-np.sum(ps * np.log(ps + 1e-12)))
    return float(np.clip(entropy / 10.0, 0.0, 1.0))


def _frame_passive_scores(face_roi_bgr: np.ndarray) -> tuple[float, float, float, float, float, tuple[float, float]]:
    gray = cv2.cvtColor(face_roi_bgr, cv2.COLOR_BGR2GRAY)
    gray128 = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)

    lap = cv2.Laplacian(gray128, cv2.CV_64F)
    texture_score = float(np.clip(lap.var() / 420.0, 0.0, 1.0))
    lbp = _lbp_score(gray128)
    moire = _moire_score(gray128)
    banding = _banding_score(gray128)
    frequency = _frequency_score(gray128)

    hsv = cv2.cvtColor(cv2.resize(face_roi_bgr, (128, 128), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    highlight_mask = (v >= 240).astype(np.uint8)
    ratio = float(np.mean(highlight_mask))
    if np.count_nonzero(highlight_mask) > 12:
        ys, xs = np.where(highlight_mask > 0)
        cx = float(np.mean(xs) / 128.0)
        cy = float(np.mean(ys) / 128.0)
    else:
        cx, cy = 0.5, 0.5

    return texture_score, lbp, moire, banding, frequency, (ratio, cx + cy)


def evaluate_passive(
    frames: List[FramePacket],
    faces: List[FaceObservation],
    cfg: PassiveLivenessConfig,
) -> tuple[bool, float, PassiveScores]:
    vals = []
    specular_trace = []

    for fp, face in zip(frames, faces):
        roi = _crop(fp.image_bgr, face.bbox_xyxy)
        if roi.size == 0:
            continue
        texture, lbp, moire, banding, frequency, spec = _frame_passive_scores(roi)
        vals.append((texture, lbp, moire, banding, frequency))
        specular_trace.append(spec)

    if not vals:
        zero = PassiveScores(0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        return False, 0.0, zero

    arr = np.asarray(vals, dtype=np.float64)
    texture = float(np.median(arr[:, 0]))
    lbp = float(np.median(arr[:, 1]))
    moire = float(np.median(arr[:, 2]))
    banding = float(np.median(arr[:, 3]))
    frequency = float(np.median(arr[:, 4]))

    spec_arr = np.asarray(specular_trace, dtype=np.float64)
    if spec_arr.shape[0] >= 3:
        ratio_std = float(np.std(spec_arr[:, 0]))
        centroid_std = float(np.std(spec_arr[:, 1]))
        specular_dynamic = float(np.clip((ratio_std * 8.0 + centroid_std * 6.0), 0.0, 1.0))
    else:
        specular_dynamic = 0.25

    passed = (
        texture >= cfg.min_texture_score
        and lbp >= cfg.min_lbp_score
        and moire <= cfg.max_moire_score
        and banding <= cfg.max_banding_score
        and frequency >= cfg.min_frequency_score
        and specular_dynamic >= cfg.min_specular_dynamic_score
    )

    # Security-oriented but less brittle confidence composition.
    # Confidence shaping: de-emphasize frequency entropy (noisy on webcams)
    # and emphasize stable bona-fide cues (texture/LBP/specular dynamics).
    conf = float(
        np.clip(
            0.28 * texture
            + 0.22 * lbp
            + 0.14 * (1.0 - moire)
            + 0.12 * (1.0 - banding)
            + 0.08 * frequency
            + 0.16 * specular_dynamic,
            0.0,
            1.0,
        )
    )
    out = PassiveScores(texture, lbp, moire, banding, frequency, specular_dynamic)
    return passed, conf, out
