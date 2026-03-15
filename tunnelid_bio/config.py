from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass(frozen=True)
class CaptureConfig:
    required_poses: List[str] = field(
        default_factory=lambda: ["center", "yaw_left", "yaw_right", "pitch_up", "pitch_down"]
    )
    burst_size: int = 12
    max_capture_seconds: float = 4.0
    pose_tolerance_deg: float = 7.5
    min_detection_confidence: float = 0.85


@dataclass(frozen=True)
class QualityConfig:
    min_sharpness: float = 90.0
    min_exposure: float = 50.0
    max_exposure: float = 205.0
    max_occlusion_ratio: float = 0.25
    max_motion_blur: float = 180.0


@dataclass(frozen=True)
class PoseBandsConfig:
    center_yaw: float = 0.0
    center_pitch: float = 0.0
    yaw_left: float = -25.0
    yaw_right: float = 25.0
    pitch_up: float = -18.0
    pitch_down: float = 18.0


@dataclass(frozen=True)
class EmbeddingConfig:
    model_type: str = "insightface"
    model_name: str = "buffalo_l"
    det_size: int = 640
    expected_dim: int = 512
    min_cosine_similarity_match: float = 0.42


@dataclass(frozen=True)
class AggregationConfig:
    min_frames_total: int = 5
    max_outlier_cosine_distance: float = 0.35
    max_intra_session_variance: float = 0.025


@dataclass(frozen=True)
class PassiveLivenessConfig:
    min_texture_score: float = 0.22
    min_lbp_score: float = 0.20
    max_moire_score: float = 0.62
    max_banding_score: float = 0.75
    min_frequency_score: float = 0.18
    min_specular_dynamic_score: float = 0.12


@dataclass(frozen=True)
class TemporalLivenessConfig:
    min_optical_flow_score: float = 0.08
    min_flow_consistency_score: float = 0.18
    min_blink_count: int = 1
    min_micro_motion_score: float = 0.03
    min_parallax_score: float = 0.05
    use_rppg: bool = False
    min_rppg_score: float = 0.04


@dataclass(frozen=True)
class ActiveLivenessConfig:
    enabled: bool = True
    prompts: List[str] = field(default_factory=lambda: ["BLINK_TWICE", "TURN_HEAD"])
    response_timeout_s: float = 3.0
    deterministic_prompt_seed: int = 2026
    blink_required_count: int = 2
    turn_min_delta_deg: float = 12.0


@dataclass(frozen=True)
class LivenessFusionConfig:
    min_overall_confidence: float = 0.55
    passive_weight: float = 0.4  # legacy field; unused in security-first fusion
    temporal_weight: float = 0.4  # legacy field; unused in security-first fusion
    active_weight: float = 0.2  # legacy field; unused in security-first fusion
    active_required: bool = True
    passive_temporal_min_confidence: float = 0.35
    replay_moire_threshold: float = 0.90
    replay_banding_threshold: float = 0.85
    replay_specular_threshold: float = 0.05
    calibration_method: str = "none"  # none|platt|isotonic
    platt_a: float = 6.0
    platt_b: float = -3.0
    isotonic_x: List[float] = field(default_factory=lambda: [0.0, 0.4, 0.6, 0.8, 1.0])
    isotonic_y: List[float] = field(default_factory=lambda: [0.0, 0.35, 0.70, 0.90, 1.0])


@dataclass(frozen=True)
class EvaluationConfig:
    default_frr_target: float = 0.05
    spoof_label: str = "spoof"


@dataclass(frozen=True)
class SecurityConfig:
    allow_embedding_plaintext_logs: bool = False
    store_raw_images: bool = False


@dataclass(frozen=True)
class VersionConfig:
    encoder: str = "arcface:buffalo_l@v1"
    preprocessing: str = "align112@v1"
    liveness: str = "passive+temporal+active@v1"
    aggregation: str = "weighted-mean-outlier@v1"


@dataclass(frozen=True)
class PipelineConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    pose_bands: PoseBandsConfig = field(default_factory=PoseBandsConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    aggregation: AggregationConfig = field(default_factory=AggregationConfig)
    passive_liveness: PassiveLivenessConfig = field(default_factory=PassiveLivenessConfig)
    temporal_liveness: TemporalLivenessConfig = field(default_factory=TemporalLivenessConfig)
    active_liveness: ActiveLivenessConfig = field(default_factory=ActiveLivenessConfig)
    liveness_fusion: LivenessFusionConfig = field(default_factory=LivenessFusionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    versions: VersionConfig = field(default_factory=VersionConfig)


def _merge_dict(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> PipelineConfig:
    cfg = PipelineConfig()
    if path is None:
        return cfg

    file_path = Path(path)
    data = yaml.safe_load(file_path.read_text()) if file_path.exists() else {}
    if not data:
        return cfg

    merged = _merge_dict(asdict(cfg), data)
    return PipelineConfig(
        capture=CaptureConfig(**merged["capture"]),
        quality=QualityConfig(**merged["quality"]),
        pose_bands=PoseBandsConfig(**merged["pose_bands"]),
        embedding=EmbeddingConfig(**merged["embedding"]),
        aggregation=AggregationConfig(**merged["aggregation"]),
        passive_liveness=PassiveLivenessConfig(**merged["passive_liveness"]),
        temporal_liveness=TemporalLivenessConfig(**merged["temporal_liveness"]),
        active_liveness=ActiveLivenessConfig(**merged["active_liveness"]),
        liveness_fusion=LivenessFusionConfig(**merged["liveness_fusion"]),
        evaluation=EvaluationConfig(**merged["evaluation"]),
        security=SecurityConfig(**merged["security"]),
        versions=VersionConfig(**merged["versions"]),
    )


def to_dict(config: PipelineConfig) -> Dict[str, Any]:
    return asdict(config)
