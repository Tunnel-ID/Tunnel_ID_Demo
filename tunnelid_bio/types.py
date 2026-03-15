from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


class SessionStatus(str, Enum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


class VerifyDecision(str, Enum):
    MATCH = "MATCH"
    NO_MATCH = "NO_MATCH"


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    image_bgr: np.ndarray
    timestamp_ms: float
    pose_hint: Optional[str] = None


@dataclass(frozen=True)
class FaceObservation:
    frame_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    landmarks: np.ndarray  # shape: (N, 2)
    detection_confidence: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


@dataclass(frozen=True)
class QualityMetrics:
    sharpness: float
    exposure: float
    occlusion_ratio: float
    motion_blur: float
    pose_error: float
    passes: bool


@dataclass(frozen=True)
class CapturedPose:
    pose_name: str
    best_frame: FramePacket
    face: FaceObservation
    quality: QualityMetrics


@dataclass
class EnrollmentResult:
    embedding: Optional[np.ndarray]
    liveness_pass: bool
    liveness_confidence: float
    quality_metrics: Dict[str, float]
    status: SessionStatus
    reason: str
    reason_codes: List[str]
    model_versions: Dict[str, str]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class VerificationResult:
    similarity_score: float
    liveness_pass: bool
    liveness_confidence: float
    decision: VerifyDecision
    reason: str
    reason_codes: List[str]
    model_versions: Dict[str, str]
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AggregationDiagnostics:
    used_count: int
    dropped_count: int
    per_frame_cosine_distance: List[float]
    intra_session_variance: float


@dataclass(frozen=True)
class LivenessResult:
    passed: bool
    confidence: float
    components: Dict[str, float]
    component_pass: Dict[str, bool]
    reason: str
    reason_codes: List[str]


@dataclass
class SessionBundle:
    session_id: str
    identity_id: str
    embedding: np.ndarray
    liveness_pass: bool
    accepted: bool
    is_spoof: bool
    auth_time_ms: float
    failure_reason: Optional[str]
