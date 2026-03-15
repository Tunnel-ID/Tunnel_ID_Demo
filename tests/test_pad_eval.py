from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tunnelid_bio.evaluation.pad_eval import run_pad_eval


def _session(root: Path, user: str, sid: str, label: str, score: float) -> None:
    d = root / "users" / user / sid
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "template.npy", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    (d / "meta.json").write_text(
        json.dumps(
            {
                "user_id": user,
                "session_id": sid,
                "label": label,
                "device_id": "cam0",
                "timestamp_epoch_s": 1.0,
                "liveness_pass": score >= 0.5,
                "liveness_confidence": score,
                "quality_score": 0.8,
                "similarity_to_enroll": 0.9,
                "lighting": "indoor",
                "distance": "near",
                "glasses": "no",
                "metadata": {"output_metadata": {"liveness_components": {"overall_confidence": score}}},
            }
        )
    )


def test_pad_eval_outputs_artifact(tmp_path: Path) -> None:
    ds = tmp_path / "pad_ds"
    out = tmp_path / "artifacts"

    _session(ds, "alice", "s1", "bona_fide", 0.9)
    _session(ds, "bob", "s1", "bona_fide", 0.8)
    _session(ds, "mallory", "s1", "attack_print", 0.2)
    _session(ds, "mallory", "s2", "attack_screen", 0.1)

    artifact = run_pad_eval(ds, out, target_bpcer=0.05)
    assert "threshold" in artifact
    assert (out / "pad_thresholds_dev.json").exists()
    assert (out / "pad_report_dev.md").exists()
