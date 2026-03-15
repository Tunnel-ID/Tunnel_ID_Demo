from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Dict, List

import numpy as np

from tunnelid_bio.capture.face_analyzer import FaceAnalyzer
from tunnelid_bio.config import ActiveLivenessConfig
from tunnelid_bio.types import FramePacket

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


@dataclass(frozen=True)
class ActiveScores:
    prompt: str
    prompt_success: bool
    prompt_score: float


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def _blink_count_from_frames(frames: List[FramePacket]) -> int:
    if mp is None:
        return 0

    def ear(arr: np.ndarray, idxs: List[int]) -> float:
        p1, p2, p3, p4, p5, p6 = arr[idxs]
        return float(
            (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
            / (2.0 * np.linalg.norm(p1 - p4) + 1e-6)
        )

    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    try:
        ears = []
        for fp in frames:
            out = mesh.process(fp.image_bgr[:, :, ::-1])
            if not out.multi_face_landmarks:
                ears.append(-1.0)
                continue
            h, w = fp.image_bgr.shape[:2]
            pts = np.array(
                [(lm.x * w, lm.y * h) for lm in out.multi_face_landmarks[0].landmark],
                dtype=np.float32,
            )
            ears.append(0.5 * (ear(pts, LEFT_EYE) + ear(pts, RIGHT_EYE)))
    finally:
        mesh.close()

    valid = np.asarray([x for x in ears if x > 0], dtype=np.float64)
    if valid.size < 5:
        return 0
    threshold = float(np.clip(np.quantile(valid, 0.35), 0.17, 0.26))

    count = 0
    closed = False
    for e in ears:
        if e < 0:
            continue
        if e < threshold and not closed:
            closed = True
        elif e >= threshold and closed:
            count += 1
            closed = False
    return count


def _yaw_series(frames: List[FramePacket]) -> List[float]:
    analyzer = FaceAnalyzer(min_detection_confidence=0.5)
    try:
        values = []
        for fp in frames:
            obs = analyzer.detect(fp.image_bgr, frame_id=fp.frame_id)
            if obs is not None:
                values.append(obs.yaw_deg)
    finally:
        analyzer.close()
    return values


def evaluate_active(
    frames: List[FramePacket],
    cfg: ActiveLivenessConfig,
) -> tuple[bool, float, ActiveScores, Dict[str, float]]:
    if not cfg.enabled:
        out = ActiveScores(prompt="DISABLED", prompt_success=True, prompt_score=1.0)
        return True, 1.0, out, {"active_enabled": 0.0}

    if not frames:
        out = ActiveScores(prompt="NONE", prompt_success=False, prompt_score=0.0)
        return False, 0.0, out, {}

    rng = Random(cfg.deterministic_prompt_seed + len(frames))
    prompt = rng.choice(cfg.prompts)

    details: Dict[str, float] = {}
    success = False
    score = 0.0

    max_frames = int(max(1, cfg.response_timeout_s * 20))
    short = frames[:max_frames]

    if prompt == "BLINK_TWICE":
        blinks = _blink_count_from_frames(short)
        details["blink_count"] = float(blinks)
        success = blinks >= cfg.blink_required_count
        score = float(np.clip(blinks / max(1, cfg.blink_required_count), 0.0, 1.0))
    elif prompt == "TURN_HEAD":
        yaws = _yaw_series(short)
        if len(yaws) >= 6:
            # Sequence-aware check: challenge must be satisfied in later segment.
            split = max(2, len(yaws) // 3)
            baseline = float(np.median(yaws[:split]))
            action = float(np.median(yaws[-split:]))
            delta = action - baseline
            details["yaw_delta"] = delta
            success = abs(delta) >= cfg.turn_min_delta_deg
            score = float(np.clip(abs(delta) / max(1.0, cfg.turn_min_delta_deg * 1.6), 0.0, 1.0))
    else:
        details["unsupported_prompt"] = 1.0

    # Replay resistance hook: session-unique timestamp signature.
    if short:
        ts = np.asarray([f.timestamp_ms for f in short], dtype=np.float64)
        jitter = float(np.std(np.diff(ts)) if ts.size > 2 else 0.0)
        details["timestamp_jitter_ms"] = jitter

    out = ActiveScores(prompt=prompt, prompt_success=success, prompt_score=score)
    return success, score, out, details
