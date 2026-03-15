from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tunnelid_bio.capture.face_analyzer import FaceAnalyzer
from tunnelid_bio.capture.quality import quality_gate, quality_weight
from tunnelid_bio.config import PipelineConfig
from tunnelid_bio.types import CapturedPose, FaceObservation, FramePacket


@dataclass
class CaptureProtocolResult:
    accepted: bool
    selected_poses: Dict[str, CapturedPose]
    rejected_frames: int
    pose_counts: Dict[str, int]
    reason: str
    metadata: Dict[str, object] = field(default_factory=dict)


class GuidedCaptureProtocol:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.required_poses = list(cfg.capture.required_poses)
        self._pose_targets = {
            "center": (cfg.pose_bands.center_yaw, cfg.pose_bands.center_pitch),
            "yaw_left": (cfg.pose_bands.yaw_left, cfg.pose_bands.center_pitch),
            "yaw_right": (cfg.pose_bands.yaw_right, cfg.pose_bands.center_pitch),
            "pitch_up": (cfg.pose_bands.center_yaw, cfg.pose_bands.pitch_up),
            "pitch_down": (cfg.pose_bands.center_yaw, cfg.pose_bands.pitch_down),
        }
        # Honor config burst_size for live usability; previous min=10 over-constrained capture.
        self._required_per_pose = int(np.clip(cfg.capture.burst_size, 1, 20))

    def _pose_match(self, pose_name: str, yaw: float, pitch: float) -> bool:
        ty, tp = self._pose_targets[pose_name]
        tol = self.cfg.capture.pose_tolerance_deg
        return abs(yaw - ty) <= tol and abs(pitch - tp) <= tol

    @staticmethod
    def _estimate_yaw_bias(observations: List[FaceObservation]) -> float:
        if not observations:
            return 0.0
        # Use the most frontal subset and estimate yaw center offset.
        frontal_rank = sorted(
            observations,
            key=lambda o: abs(o.yaw_deg) + 0.5 * abs(o.pitch_deg),
        )
        k = max(5, int(len(frontal_rank) * 0.25))
        selected = frontal_rank[: min(k, len(frontal_rank))]
        return float(np.median([o.yaw_deg for o in selected]))

    @staticmethod
    def _estimate_yaw_sign(detected: List[Tuple[FramePacket, FaceObservation]]) -> float:
        left_vals = [face.yaw_deg for fp, face in detected if fp.pose_hint == "yaw_left"]
        right_vals = [face.yaw_deg for fp, face in detected if fp.pose_hint == "yaw_right"]
        if len(left_vals) < 3 or len(right_vals) < 3:
            return 1.0
        left_m = float(np.median(left_vals))
        right_m = float(np.median(right_vals))
        # Expected convention: yaw_left < yaw_right.
        return 1.0 if left_m < right_m else -1.0

    def run(self, frames_sequence: List[FramePacket]) -> CaptureProtocolResult:
        if not frames_sequence:
            return CaptureProtocolResult(
                accepted=False,
                selected_poses={},
                rejected_frames=0,
                pose_counts={p: 0 for p in self.required_poses},
                reason="NO_FRAMES",
            )

        pose_candidates: Dict[str, List[CapturedPose]] = {p: [] for p in self.required_poses}
        rejected_frames = 0
        analyzer = FaceAnalyzer(min_detection_confidence=self.cfg.capture.min_detection_confidence)
        start_ts = float(frames_sequence[0].timestamp_ms)
        # Guided captures often include a per-pose sequence that exceeds legacy max_capture_seconds.
        # If pose hints exist, trust the recorded sequence and don't truncate early.
        has_pose_hints = any(fp.pose_hint is not None for fp in frames_sequence)
        allowed_window_s = float("inf") if has_pose_hints else float(self.cfg.capture.max_capture_seconds)
        detected: List[Tuple[FramePacket, FaceObservation]] = []

        try:
            for fp in frames_sequence:
                if (fp.timestamp_ms - start_ts) / 1000.0 > allowed_window_s:
                    break
                face = analyzer.detect(fp.image_bgr, frame_id=fp.frame_id)
                if face is None:
                    rejected_frames += 1
                    continue
                detected.append((fp, face))

            yaw_bias = self._estimate_yaw_bias([f for _, f in detected])
            yaw_sign = self._estimate_yaw_sign(detected)

            for fp, raw_face in detected:
                adj_face = FaceObservation(
                    frame_id=raw_face.frame_id,
                    bbox_xyxy=raw_face.bbox_xyxy,
                    landmarks=raw_face.landmarks,
                    detection_confidence=raw_face.detection_confidence,
                    yaw_deg=float((raw_face.yaw_deg - yaw_bias) * yaw_sign),
                    pitch_deg=raw_face.pitch_deg,
                    roll_deg=raw_face.roll_deg,
                )

                matched_any = False
                for pose_name in self.required_poses:
                    if not self._pose_match(pose_name, adj_face.yaw_deg, adj_face.pitch_deg):
                        continue
                    matched_any = True
                    qm = quality_gate(fp.image_bgr, adj_face, pose_name, self.cfg)
                    if not qm.passes:
                        rejected_frames += 1
                        continue
                    pose_candidates[pose_name].append(
                        CapturedPose(
                            pose_name=pose_name,
                            best_frame=fp,
                            face=adj_face,
                            quality=qm,
                        )
                    )
                if not matched_any:
                    rejected_frames += 1
        finally:
            analyzer.close()

        pose_counts = {p: len(v) for p, v in pose_candidates.items()}
        missing = [p for p in self.required_poses if len(pose_candidates[p]) < self._required_per_pose]
        if missing:
            return CaptureProtocolResult(
                accepted=False,
                selected_poses={},
                rejected_frames=rejected_frames,
                pose_counts=pose_counts,
                reason=f"INSUFFICIENT_POSE_COVERAGE:{','.join(missing)}",
                metadata={
                    "required_per_pose": self._required_per_pose,
                    "yaw_bias_deg": yaw_bias,
                    "yaw_sign": yaw_sign,
                },
            )

        selected: Dict[str, CapturedPose] = {}
        for pose_name, candidates in pose_candidates.items():
            best = max(candidates, key=lambda c: quality_weight(c.quality))
            selected[pose_name] = best

        return CaptureProtocolResult(
            accepted=True,
            selected_poses=selected,
            rejected_frames=rejected_frames,
            pose_counts=pose_counts,
            reason="OK",
            metadata={
                "required_per_pose": self._required_per_pose,
                "yaw_bias_deg": yaw_bias,
                "yaw_sign": yaw_sign,
            },
        )

    def get_prompt_order(self) -> List[str]:
        return self.required_poses.copy()

    def next_missing_pose(self, pose_counts: Dict[str, int]) -> Optional[str]:
        for pose in self.required_poses:
            if pose_counts.get(pose, 0) < self._required_per_pose:
                return pose
        return None
