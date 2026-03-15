from __future__ import annotations

from tunnelid_bio.config import PipelineConfig
from tunnelid_bio.liveness.fusion import LivenessEngine


class _PS:
    def __init__(self, p):
        self.texture_score = p.get("texture", 0.5)
        self.lbp_score = p.get("lbp", 0.5)
        self.moire_score = p.get("moire", 0.1)
        self.banding_score = p.get("banding", 0.1)
        self.frequency_score = p.get("freq", 0.5)
        self.specular_dynamic_score = p.get("spec", 0.3)


class _TS:
    def __init__(self, t):
        self.optical_flow_score = t.get("flow", 0.5)
        self.flow_consistency_score = t.get("flowc", 0.6)
        self.blink_count = t.get("blinks", 2)
        self.micro_motion_score = t.get("micro", 0.5)
        self.parallax_score = t.get("parallax", 0.3)
        self.rppg_score = t.get("rppg", 0.0)


class _AS:
    def __init__(self, prompt_score=1.0):
        self.prompt = "BLINK_TWICE"
        self.prompt_success = True
        self.prompt_score = prompt_score


def test_fusion_uses_active_with_max_pt(monkeypatch):
    cfg = PipelineConfig()
    eng = LivenessEngine(cfg)

    def fake_detect(self, frames):
        return [], []

    def fake_passive(frames, faces, cfg):
        return False, 0.2, _PS({"moire": 0.3, "banding": 0.2, "spec": 0.4})

    def fake_temporal(frames, cfg):
        return True, 0.8, _TS({})

    def fake_active(frames, cfg):
        return True, 1.0, _AS(1.0), {}

    monkeypatch.setattr(LivenessEngine, "_detect_faces_for_passive", fake_detect)
    monkeypatch.setattr("tunnelid_bio.liveness.fusion.evaluate_passive", fake_passive)
    monkeypatch.setattr("tunnelid_bio.liveness.fusion.evaluate_temporal", fake_temporal)
    monkeypatch.setattr("tunnelid_bio.liveness.fusion.evaluate_active", fake_active)

    out = eng.evaluate([])
    assert out.passed
    assert out.confidence >= 0.79
    assert out.components["overall_raw_confidence"] == 0.8


def test_fusion_replay_suspect_blocks(monkeypatch):
    cfg = PipelineConfig()
    eng = LivenessEngine(cfg)

    monkeypatch.setattr(LivenessEngine, "_detect_faces_for_passive", lambda self, frames: ([], []))
    monkeypatch.setattr(
        "tunnelid_bio.liveness.fusion.evaluate_passive",
        lambda frames, faces, cfg: (True, 0.9, _PS({"moire": 0.99, "banding": 0.1, "spec": 0.3})),
    )
    monkeypatch.setattr("tunnelid_bio.liveness.fusion.evaluate_temporal", lambda frames, cfg: (True, 0.9, _TS({})))
    monkeypatch.setattr("tunnelid_bio.liveness.fusion.evaluate_active", lambda frames, cfg: (True, 0.9, _AS(0.9), {}))

    out = eng.evaluate([])
    assert not out.passed
    assert "LIVENESS_FAIL_REPLAY_SUSPECT" in out.reason_codes
