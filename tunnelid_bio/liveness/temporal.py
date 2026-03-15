from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from tunnelid_bio.capture.face_analyzer import FaceAnalyzer
from tunnelid_bio.config import TemporalLivenessConfig
from tunnelid_bio.types import FramePacket

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP_IDX = 1
LEFT_CHEEK_IDX = 234
RIGHT_CHEEK_IDX = 454
FOREHEAD_IDX = 10


@dataclass(frozen=True)
class TemporalScores:
    optical_flow_score: float
    flow_consistency_score: float
    blink_count: int
    micro_motion_score: float
    parallax_score: float
    rppg_score: float


def _ear(pts: np.ndarray) -> float:
    p1, p2, p3, p4, p5, p6 = pts
    num = np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)
    den = 2.0 * np.linalg.norm(p1 - p4) + 1e-6
    return float(num / den)


def _extract_landmarks(frame_bgr: np.ndarray, mesh) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    out = mesh.process(frame_bgr[:, :, ::-1])
    if not out.multi_face_landmarks:
        z = np.zeros(2, dtype=np.float32)
        return -1.0, z, z, z

    face_lm = out.multi_face_landmarks[0].landmark
    h, w = frame_bgr.shape[:2]
    arr = np.array([(lm.x * w, lm.y * h) for lm in face_lm], dtype=np.float32)
    ear = 0.5 * (_ear(arr[LEFT_EYE]) + _ear(arr[RIGHT_EYE]))
    nose = arr[NOSE_TIP_IDX]
    cheeks = 0.5 * (arr[LEFT_CHEEK_IDX] + arr[RIGHT_CHEEK_IDX])
    forehead = arr[FOREHEAD_IDX]
    return ear, nose, cheeks, forehead


def _blink_count(ear_series: List[float]) -> int:
    valid = np.asarray([x for x in ear_series if x > 0], dtype=np.float64)
    if valid.size < 5:
        return 0

    low = float(np.quantile(valid, 0.20))
    high = float(np.quantile(valid, 0.80))
    threshold = float(np.clip((low + high) * 0.5, 0.17, 0.27))

    count = 0
    closed = False
    closed_len = 0
    for ear in ear_series:
        if ear < 0:
            continue
        if ear < threshold:
            closed_len += 1
            if not closed:
                closed = True
        elif closed:
            if 1 <= closed_len <= 8:
                count += 1
            closed = False
            closed_len = 0
    return count


def _face_optical_flow(frames: List[FramePacket]) -> Tuple[float, float]:
    if len(frames) < 2:
        return 0.0, 0.0

    analyzer = FaceAnalyzer(min_detection_confidence=0.5)
    mags = []
    try:
        prev_gray = None
        prev_bbox = None
        for fp in frames:
            obs = analyzer.detect(fp.image_bgr, frame_id=fp.frame_id)
            gray = cv2.cvtColor(fp.image_bgr, cv2.COLOR_BGR2GRAY)
            if obs is None:
                prev_gray = gray
                prev_bbox = None
                continue
            x1, y1, x2, y2 = obs.bbox_xyxy
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

            if prev_gray is None or prev_bbox is None:
                prev_gray = gray
                prev_bbox = (x1, y1, x2, y2)
                continue

            px1, py1, px2, py2 = prev_bbox
            rx1, ry1 = max(x1, px1), max(y1, py1)
            rx2, ry2 = min(x2, px2), min(y2, py2)
            if rx2 - rx1 < 24 or ry2 - ry1 < 24:
                prev_gray = gray
                prev_bbox = (x1, y1, x2, y2)
                continue

            prev_roi = prev_gray[ry1:ry2, rx1:rx2]
            curr_roi = gray[ry1:ry2, rx1:rx2]
            flow = cv2.calcOpticalFlowFarneback(prev_roi, curr_roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mags.append(float(np.median(mag)))

            prev_gray = gray
            prev_bbox = (x1, y1, x2, y2)
    finally:
        analyzer.close()

    if not mags:
        return 0.0, 0.0

    arr = np.asarray(mags, dtype=np.float64)
    mag_score = float(np.clip(np.median(arr) / 1.8, 0.0, 1.0))
    cv = float(np.std(arr) / (np.mean(arr) + 1e-6))
    consistency = float(np.clip(1.0 - cv, 0.0, 1.0))
    return mag_score, consistency


def _rppg_score(signal: np.ndarray, fps: float = 20.0) -> float:
    if signal.size < int(max(20, fps * 2.0)):
        return 0.0
    x = signal.astype(np.float64)
    x = x - np.mean(x)
    if float(np.std(x)) < 1e-6:
        return 0.0

    win = np.hanning(x.size)
    xf = np.fft.rfft(x * win)
    power = np.abs(xf) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fps)
    band = (freqs >= 0.8) & (freqs <= 2.5)
    if not np.any(band):
        return 0.0

    p_band = power[band]
    if p_band.size < 3:
        return 0.0

    peak = float(np.max(p_band))
    med = float(np.median(p_band) + 1e-6)
    return float(np.clip((peak / med - 1.0) / 8.0, 0.0, 1.0))


def evaluate_temporal(
    frames: List[FramePacket],
    cfg: TemporalLivenessConfig,
) -> tuple[bool, float, TemporalScores]:
    flow_score, flow_consistency = _face_optical_flow(frames)

    blink_count = 0
    micro_motion = 0.0
    parallax = 0.0
    rppg = 0.0

    if mp is not None:
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            ears = []
            noses = []
            cheeks = []
            forehead_pts = []
            cheek_green = []
            ts = []

            for fp in frames:
                ear, nose, cheek, forehead = _extract_landmarks(fp.image_bgr, mesh)
                ears.append(ear)
                if ear >= 0:
                    noses.append(nose)
                    cheeks.append(cheek)
                    forehead_pts.append(forehead)

                    h, w = fp.image_bgr.shape[:2]
                    cx, cy = int(np.clip(cheek[0], 0, w - 1)), int(np.clip(cheek[1], 0, h - 1))
                    r = 8
                    patch = fp.image_bgr[max(0, cy - r) : min(h, cy + r), max(0, cx - r) : min(w, cx + r)]
                    if patch.size:
                        cheek_green.append(float(np.mean(patch[:, :, 1])))
                        ts.append(float(fp.timestamp_ms))

            blink_count = _blink_count(ears)

            if len(noses) >= 2:
                nose_arr = np.asarray(noses, dtype=np.float64)
                disp = np.linalg.norm(np.diff(nose_arr, axis=0), axis=1)
                micro_motion = float(np.clip(np.median(disp) / 6.0, 0.0, 1.0))

            if len(noses) >= 3 and len(cheeks) >= 3:
                nose_disp = np.linalg.norm(np.diff(np.asarray(noses), axis=0), axis=1)
                cheek_disp = np.linalg.norm(np.diff(np.asarray(cheeks), axis=0), axis=1)
                diff = np.abs(nose_disp - cheek_disp)
                parallax = float(np.clip(np.median(diff) / 3.5, 0.0, 1.0))

            if cfg.use_rppg and len(cheek_green) >= 20:
                dt_ms = np.diff(np.asarray(ts, dtype=np.float64))
                fps_est = 1000.0 / float(np.median(dt_ms)) if dt_ms.size else 20.0
                rppg = _rppg_score(np.asarray(cheek_green, dtype=np.float64), fps=float(np.clip(fps_est, 10.0, 30.0)))
        finally:
            mesh.close()

    passes_flow = flow_score >= cfg.min_optical_flow_score and flow_consistency >= cfg.min_flow_consistency_score
    passes_blink = blink_count >= cfg.min_blink_count
    passes_motion = micro_motion >= cfg.min_micro_motion_score and parallax >= cfg.min_parallax_score
    passes_rppg = (not cfg.use_rppg) or (rppg >= cfg.min_rppg_score)

    passed = passes_flow and passes_blink and passes_motion and passes_rppg

    blink_norm = float(np.clip(blink_count / 2.0, 0.0, 1.0))
    # Confidence shaping: flow consistency is often unstable with webcam AE/flicker.
    conf = float(
        np.clip(
            0.24 * flow_score
            + 0.08 * flow_consistency
            + 0.30 * blink_norm
            + 0.22 * micro_motion
            + 0.12 * parallax
            + 0.04 * rppg,
            0.0,
            1.0,
        )
    )

    scores = TemporalScores(
        optical_flow_score=flow_score,
        flow_consistency_score=flow_consistency,
        blink_count=blink_count,
        micro_motion_score=micro_motion,
        parallax_score=parallax,
        rppg_score=rppg,
    )
    return passed, conf, scores
