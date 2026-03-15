from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from tunnelid_bio.aggregation.aggregator import MultiFrameAggregator
from tunnelid_bio.capture.protocol import GuidedCaptureProtocol
from tunnelid_bio.capture.quality import quality_weight
from tunnelid_bio.config import PipelineConfig, load_config
from tunnelid_bio.embedding.encoder import ArcFaceEncoder
from tunnelid_bio.liveness.fusion import LivenessEngine
from tunnelid_bio.metrics.drift import drift_risk
from tunnelid_bio.quality.quality_score import quality_gate, quality_score, summarize_quality
from tunnelid_bio.types import (
    EnrollmentResult,
    FramePacket,
    QualityMetrics,
    SessionStatus,
    VerificationResult,
    VerifyDecision,
)
from tunnelid_bio.utils.math_utils import cosine_distance, cosine_similarity, l2_normalize
from tunnelid_bio.versioning import get_version_tags


@dataclass
class BiometricPipeline:
    cfg: PipelineConfig

    def __post_init__(self) -> None:
        np.random.seed(2026)
        cv2.setNumThreads(1)

        self.capture = GuidedCaptureProtocol(self.cfg)
        self.encoder = ArcFaceEncoder(self.cfg.embedding)
        self.aggregator = MultiFrameAggregator(self.cfg.aggregation)
        self.liveness = LivenessEngine(self.cfg)
        self.version_tags = get_version_tags(self.cfg)

    @classmethod
    def from_config_path(cls, config_path: str | None = None) -> "BiometricPipeline":
        return cls(load_config(config_path))

    @staticmethod
    def _normalize_frames(frames_sequence: Sequence[np.ndarray | FramePacket]) -> List[FramePacket]:
        out = []
        now = time.time() * 1000.0
        for i, frame in enumerate(frames_sequence):
            if isinstance(frame, FramePacket):
                out.append(frame)
            elif isinstance(frame, np.ndarray):
                out.append(FramePacket(frame_id=i, image_bgr=frame, timestamp_ms=now + i * 33.0))
            else:
                raise TypeError(f"Unsupported frame type: {type(frame)!r}")
        return out

    @staticmethod
    def _map_capture_reason(reason: str) -> str:
        if reason.startswith("INSUFFICIENT_POSE_COVERAGE"):
            return "REJECT_POSE"
        if reason == "NO_FRAMES":
            return "REJECT_NO_FACE"
        return "REJECT_CAPTURE"

    def _extract_templates(
        self,
        cap,
    ) -> tuple[np.ndarray, object, List[QualityMetrics], List[float]]:
        embeddings = []
        qweights = []
        pose_metrics: List[QualityMetrics] = []
        for pose_name in self.cfg.capture.required_poses:
            picked = cap.selected_poses[pose_name]
            emb = self.encoder.encode(picked.best_frame.image_bgr, face_hint=picked.face)
            embeddings.append(emb)
            qweights.append(quality_weight(picked.quality))
            pose_metrics.append(picked.quality)

        template, diag = self.aggregator.aggregate(embeddings, qweights)
        return template, diag, pose_metrics, qweights

    def _resolve_runtime_thresholds(self, thresholds: Optional[Dict[str, object]]) -> Dict[str, float | str]:
        rec = dict((thresholds or {}).get("recommended", {})) if thresholds else {}
        return {
            "tau_face": float(rec.get("tau_face", self.cfg.embedding.min_cosine_similarity_match)),
            "quality_gate_threshold": float(rec.get("quality_gate_threshold", 0.0)),
            "drift_bound_delta": float(rec.get("drift_bound_delta", 0.0)),
            "match_strategy": str(rec.get("match_strategy", "nearest-template")),
        }

    def _version_compatible(self, enrolled_versions: Optional[Dict[str, str]], allow_mismatch: bool) -> bool:
        if allow_mismatch or not enrolled_versions:
            return True
        keys = ("encoder", "preprocessing", "liveness", "aggregation")
        for k in keys:
            if enrolled_versions.get(k) != self.version_tags.get(k):
                return False
        return True

    @staticmethod
    def _inter_template_distance(stored_embeddings: np.ndarray) -> float:
        if stored_embeddings.ndim == 1 or stored_embeddings.shape[0] < 2:
            return 0.0
        dists = []
        for i in range(stored_embeddings.shape[0]):
            for j in range(i + 1, stored_embeddings.shape[0]):
                dists.append(cosine_distance(stored_embeddings[i], stored_embeddings[j]))
        return float(np.mean(np.asarray(dists, dtype=np.float64))) if dists else 0.0

    def enroll_session(
        self,
        frames_sequence: Sequence[np.ndarray | FramePacket],
        enforce_liveness: bool = True,
        thresholds: Optional[Dict[str, object]] = None,
        existing_templates: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        start = time.perf_counter()
        frames = self._normalize_frames(frames_sequence)
        runtime_thr = self._resolve_runtime_thresholds(thresholds)

        cap = self.capture.run(frames)
        if not cap.accepted:
            elapsed = (time.perf_counter() - start) * 1000.0
            code = self._map_capture_reason(cap.reason)
            res = EnrollmentResult(
                embedding=None,
                liveness_pass=False,
                liveness_confidence=0.0,
                quality_metrics={"capture_to_decision_ms": elapsed},
                status=SessionStatus.REJECTED,
                reason=code,
                reason_codes=[code],
                model_versions=self.version_tags,
                metadata={"pose_counts": cap.pose_counts, "capture": cap.metadata},
            )
            return self._to_dict_enroll(res)

        try:
            template, diag, pose_metrics, _ = self._extract_templates(cap)
        except ValueError as exc:
            elapsed = (time.perf_counter() - start) * 1000.0
            code = "REJECT_DRIFT_RISK"
            res = EnrollmentResult(
                embedding=None,
                liveness_pass=False,
                liveness_confidence=0.0,
                quality_metrics={"capture_to_decision_ms": elapsed},
                status=SessionStatus.REJECTED,
                reason=code,
                reason_codes=[code, f"AGGREGATION_FAIL:{exc}"],
                model_versions=self.version_tags,
                metadata={"pose_counts": cap.pose_counts},
            )
            return self._to_dict_enroll(res)

        liveness = self.liveness.evaluate(frames)
        liveness_ok = liveness.passed if enforce_liveness else True
        q_score = quality_score(
            pose_metrics,
            liveness_confidence=(liveness.confidence if enforce_liveness else 1.0),
            intra_session_variance=diag.intra_session_variance,
        )
        quality_ok, _ = quality_gate(q_score, float(runtime_thr["quality_gate_threshold"]))

        reason_codes: List[str] = []
        if not liveness_ok:
            reason_codes.append("REJECT_LIVENESS")
            reason_codes.extend([f"REJECT_{rc}" for rc in liveness.reason_codes])
        if not quality_ok:
            reason_codes.append("REJECT_LOW_QUALITY_SCORE")
        if diag.intra_session_variance > self.cfg.aggregation.max_intra_session_variance:
            reason_codes.append("REJECT_DRIFT_RISK")

        accepted = not reason_codes
        elapsed = (time.perf_counter() - start) * 1000.0

        merged_templates = None
        if existing_templates is not None:
            arr = np.asarray(existing_templates, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            merged_templates = np.vstack([arr, template.reshape(1, -1)])

        res = EnrollmentResult(
            embedding=template if accepted else None,
            liveness_pass=liveness_ok,
            liveness_confidence=liveness.confidence if enforce_liveness else 1.0,
            quality_metrics={
                "intra_session_variance": diag.intra_session_variance,
                "mean_pose_error": float(np.mean(np.asarray([m.pose_error for m in pose_metrics], dtype=np.float64))),
                "capture_to_decision_ms": elapsed,
                "used_frames": float(diag.used_count),
                "dropped_frames": float(diag.dropped_count),
                "quality_score": q_score,
            },
            status=SessionStatus.ACCEPTED if accepted else SessionStatus.REJECTED,
            reason="OK" if accepted else reason_codes[0],
            reason_codes=([] if accepted else reason_codes),
            model_versions=self.version_tags,
            metadata={
                "score_type": "cosine",
                "score_normalization": "raw",
                "quality_gate": {
                    "threshold": float(runtime_thr["quality_gate_threshold"]),
                    "passed": quality_ok,
                },
                "pose_counts": cap.pose_counts,
                "per_frame_distances": diag.per_frame_cosine_distance,
                "drift_metrics": {
                    "mean_intra_session_distance": float(np.mean(np.asarray(diag.per_frame_cosine_distance, dtype=np.float64))),
                    "max_intra_session_distance": float(np.max(np.asarray(diag.per_frame_cosine_distance, dtype=np.float64))),
                    "inter_template_distance": self._inter_template_distance(merged_templates)
                    if merged_templates is not None
                    else 0.0,
                    "drift_risk": bool(diag.intra_session_variance > self.cfg.aggregation.max_intra_session_variance),
                },
                "quality_components": summarize_quality(pose_metrics),
                "liveness_components": liveness.components,
                "liveness_component_pass": liveness.component_pass,
                "liveness_reason_codes": liveness.reason_codes,
                "liveness_enforced": enforce_liveness,
                "thresholds_used": runtime_thr,
            },
        )
        return self._to_dict_enroll(res)

    def verify_session(
        self,
        frames_sequence: Sequence[np.ndarray | FramePacket],
        stored_embedding: np.ndarray,
        enforce_liveness: bool = True,
        thresholds: Optional[Dict[str, object]] = None,
        enrolled_model_versions: Optional[Dict[str, str]] = None,
        allow_model_mismatch: bool = False,
    ) -> Dict[str, object]:
        start = time.perf_counter()
        frames = self._normalize_frames(frames_sequence)
        runtime_thr = self._resolve_runtime_thresholds(thresholds)

        cap = self.capture.run(frames)
        if not cap.accepted:
            code = self._map_capture_reason(cap.reason)
            res = VerificationResult(
                similarity_score=0.0,
                liveness_pass=False,
                liveness_confidence=0.0,
                decision=VerifyDecision.NO_MATCH,
                reason=code,
                reason_codes=[code],
                model_versions=self.version_tags,
                metadata={"pose_counts": cap.pose_counts},
            )
            return self._to_dict_verify(res)

        try:
            template, diag, pose_metrics, _ = self._extract_templates(cap)
        except ValueError as exc:
            code = "REJECT_DRIFT_RISK"
            res = VerificationResult(
                similarity_score=0.0,
                liveness_pass=False,
                liveness_confidence=0.0,
                decision=VerifyDecision.NO_MATCH,
                reason=code,
                reason_codes=[code, f"AGGREGATION_FAIL:{exc}"],
                model_versions=self.version_tags,
            )
            return self._to_dict_verify(res)

        stored = np.asarray(stored_embedding, dtype=np.float32)
        if stored.ndim == 1:
            stored = stored.reshape(1, -1)

        similarity_per_template = [
            cosine_similarity(l2_normalize(template), l2_normalize(stored[i]))
            for i in range(stored.shape[0])
        ]
        best_similarity = float(np.max(np.asarray(similarity_per_template, dtype=np.float64)))
        mean_topk_similarity = float(np.mean(np.sort(np.asarray(similarity_per_template))[-min(3, len(similarity_per_template)) :]))
        similarity = (
            mean_topk_similarity
            if str(runtime_thr["match_strategy"]) == "mean-topk"
            else best_similarity
        )

        liveness = self.liveness.evaluate(frames)
        liveness_ok = liveness.passed if enforce_liveness else True
        version_ok = self._version_compatible(enrolled_model_versions, allow_model_mismatch)

        q_score = quality_score(
            pose_metrics,
            liveness_confidence=(liveness.confidence if enforce_liveness else 1.0),
            intra_session_variance=diag.intra_session_variance,
        )
        quality_ok, _ = quality_gate(q_score, float(runtime_thr["quality_gate_threshold"]))

        drift_distance = float(1.0 - best_similarity)
        drift_bound = float(runtime_thr["drift_bound_delta"])
        drift_flag = drift_risk(drift_distance, drift_bound) if drift_bound > 0.0 else False

        reason_codes: List[str] = []
        if not version_ok:
            reason_codes.append("REJECT_MODEL_VERSION_MISMATCH")
        if not liveness_ok:
            reason_codes.append("REJECT_LIVENESS")
            reason_codes.extend([f"REJECT_{rc}" for rc in liveness.reason_codes])
        if not quality_ok:
            reason_codes.append("REJECT_LOW_QUALITY_SCORE")
        if drift_flag:
            reason_codes.append("REJECT_DRIFT_RISK")
        if similarity < float(runtime_thr["tau_face"]):
            reason_codes.append("REJECT_SIMILARITY_BELOW_THRESHOLD")

        decision = VerifyDecision.MATCH if not reason_codes else VerifyDecision.NO_MATCH
        elapsed = (time.perf_counter() - start) * 1000.0

        res = VerificationResult(
            similarity_score=float(similarity),
            liveness_pass=liveness_ok,
            liveness_confidence=liveness.confidence if enforce_liveness else 1.0,
            decision=decision,
            reason=("OK" if decision == VerifyDecision.MATCH else reason_codes[0]),
            reason_codes=([] if decision == VerifyDecision.MATCH else reason_codes),
            model_versions=self.version_tags,
            metadata={
                "capture_to_decision_ms": elapsed,
                "score_type": "cosine",
                "score_normalization": "raw",
                "thresholds_used": runtime_thr,
                "quality_score": q_score,
                "quality_gate": {
                    "threshold": float(runtime_thr["quality_gate_threshold"]),
                    "passed": quality_ok,
                },
                "similarity_per_template": similarity_per_template,
                "match_strategy": str(runtime_thr["match_strategy"]),
                "drift_metrics": {
                    "mean_intra_session_distance": float(np.mean(np.asarray(diag.per_frame_cosine_distance, dtype=np.float64))),
                    "max_intra_session_distance": float(np.max(np.asarray(diag.per_frame_cosine_distance, dtype=np.float64))),
                    "inter_template_distance": self._inter_template_distance(stored),
                    "drift_distance": drift_distance,
                    "drift_bound": drift_bound,
                    "drift_risk": drift_flag,
                },
                "quality_components": summarize_quality(pose_metrics),
                "liveness_components": liveness.components,
                "liveness_component_pass": liveness.component_pass,
                "liveness_reason_codes": liveness.reason_codes,
                "liveness_enforced": enforce_liveness,
                "model_version_compatible": version_ok,
            },
        )
        return self._to_dict_verify(res)

    @staticmethod
    def _to_dict_enroll(r: EnrollmentResult) -> Dict[str, object]:
        return {
            "embedding": r.embedding,
            "liveness_pass": r.liveness_pass,
            "liveness_confidence": r.liveness_confidence,
            "quality_metrics": r.quality_metrics,
            "status": r.status.value,
            "reason": r.reason,
            "reason_codes": r.reason_codes,
            "model_versions": r.model_versions,
            "metadata": r.metadata,
        }

    @staticmethod
    def _to_dict_verify(r: VerificationResult) -> Dict[str, object]:
        return {
            "similarity_score": r.similarity_score,
            "liveness_pass": r.liveness_pass,
            "liveness_confidence": r.liveness_confidence,
            "decision": r.decision.value,
            "reason": r.reason,
            "reason_codes": r.reason_codes,
            "model_versions": r.model_versions,
            "metadata": r.metadata,
        }
