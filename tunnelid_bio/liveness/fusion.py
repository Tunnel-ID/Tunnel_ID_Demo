from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from tunnelid_bio.capture.face_analyzer import FaceAnalyzer
from tunnelid_bio.config import PipelineConfig
from tunnelid_bio.liveness.active import evaluate_active
from tunnelid_bio.liveness.passive import evaluate_passive
from tunnelid_bio.liveness.temporal import evaluate_temporal
from tunnelid_bio.types import FramePacket, LivenessResult


@dataclass
class LivenessEngine:
    cfg: PipelineConfig

    def _detect_faces_for_passive(self, frames: List[FramePacket]):
        analyzer = FaceAnalyzer(min_detection_confidence=0.5)
        try:
            matched_frames = []
            faces = []
            for fp in frames:
                obs = analyzer.detect(fp.image_bgr, frame_id=fp.frame_id)
                if obs is not None:
                    matched_frames.append(fp)
                    faces.append(obs)
        finally:
            analyzer.close()
        return matched_frames, faces

    def _calibrate(self, raw_score: float) -> float:
        fcfg = self.cfg.liveness_fusion
        method = str(fcfg.calibration_method).lower()
        x = float(np.clip(raw_score, 0.0, 1.0))

        if method == "platt":
            z = fcfg.platt_a * x + fcfg.platt_b
            return float(1.0 / (1.0 + np.exp(-z)))

        if method == "isotonic":
            xp = np.asarray(list(fcfg.isotonic_x), dtype=np.float64)
            yp = np.asarray(list(fcfg.isotonic_y), dtype=np.float64)
            if xp.size < 2 or yp.size != xp.size:
                return x
            xp = np.clip(xp, 0.0, 1.0)
            yp = np.clip(yp, 0.0, 1.0)
            order = np.argsort(xp)
            xp = xp[order]
            yp = yp[order]
            return float(np.interp(x, xp, yp))

        return x

    def evaluate(self, frames: List[FramePacket]) -> LivenessResult:
        matched_frames, faces = self._detect_faces_for_passive(frames)

        p_pass, p_conf, p_scores = evaluate_passive(
            matched_frames,
            faces,
            self.cfg.passive_liveness,
        )
        t_pass, t_conf, t_scores = evaluate_temporal(frames, self.cfg.temporal_liveness)
        a_pass, a_conf, a_scores, a_details = evaluate_active(frames, self.cfg.active_liveness)

        fcfg = self.cfg.liveness_fusion
        pt_conf = max(float(p_conf), float(t_conf))
        overall_raw = min(float(a_conf), pt_conf) if fcfg.active_required else pt_conf

        replay_suspect = (
            p_scores.moire_score >= fcfg.replay_moire_threshold
            or p_scores.banding_score >= fcfg.replay_banding_threshold
            or p_scores.specular_dynamic_score <= fcfg.replay_specular_threshold
        )

        hard_reasons: List[str] = []
        advisory_reasons: List[str] = []
        if fcfg.active_required and not a_pass:
            hard_reasons.append("LIVENESS_FAIL_ACTIVE")
        if (not p_pass) and (not t_pass):
            advisory_reasons.append("LIVENESS_FAIL_PASSIVE")
            advisory_reasons.append("LIVENESS_FAIL_TEMPORAL")
        if replay_suspect:
            hard_reasons.append("LIVENESS_FAIL_REPLAY_SUSPECT")
        if pt_conf < fcfg.passive_temporal_min_confidence:
            hard_reasons.append("LIVENESS_FAIL_PT_MIN")

        calibrated_conf = self._calibrate(overall_raw)
        if calibrated_conf < fcfg.min_overall_confidence:
            hard_reasons.append("LIVENESS_FAIL_LOW_CONFIDENCE")

        passed = len(hard_reasons) == 0

        components: Dict[str, float] = {
            "passive_confidence": p_conf,
            "temporal_confidence": t_conf,
            "active_confidence": a_conf,
            "passive_texture": p_scores.texture_score,
            "passive_lbp": p_scores.lbp_score,
            "passive_moire": p_scores.moire_score,
            "passive_banding": p_scores.banding_score,
            "passive_frequency": p_scores.frequency_score,
            "passive_specular_dynamic": p_scores.specular_dynamic_score,
            "temporal_optical_flow": t_scores.optical_flow_score,
            "temporal_flow_consistency": t_scores.flow_consistency_score,
            "temporal_blink_count": float(t_scores.blink_count),
            "temporal_micro_motion": t_scores.micro_motion_score,
            "temporal_parallax": t_scores.parallax_score,
            "temporal_rppg": t_scores.rppg_score,
            "active_prompt_score": a_scores.prompt_score,
            "pt_confidence": pt_conf,
            "overall_raw_confidence": overall_raw,
            "overall_confidence": calibrated_conf,
            "replay_suspect": float(replay_suspect),
        }
        components.update({f"active_{k}": v for k, v in a_details.items()})

        comp_pass = {
            "passive": bool(p_pass),
            "temporal": bool(t_pass),
            "active": bool(a_pass),
            "replay_suspect": not replay_suspect,
            "pt_min": bool(pt_conf >= fcfg.passive_temporal_min_confidence),
        }

        reason_codes = hard_reasons + advisory_reasons
        reason = "OK" if passed else hard_reasons[0]
        return LivenessResult(
            passed=passed,
            confidence=float(calibrated_conf),
            components=components,
            component_pass=comp_pass,
            reason=reason,
            reason_codes=([] if passed else reason_codes),
        )
