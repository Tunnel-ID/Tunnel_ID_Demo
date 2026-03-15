from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tunnelid_bio.evaluation.runner import evaluate_dataset


def _write_session(root: Path, user: str, sess: str, label: str, emb: np.ndarray, liveness: bool, quality: float) -> None:
    sdir = root / "users" / user / sess
    sdir.mkdir(parents=True, exist_ok=True)
    np.save(sdir / "template.npy", emb.astype(np.float32))
    (sdir / "meta.json").write_text(
        json.dumps(
            {
                "user_id": user,
                "session_id": sess,
                "label": label,
                "device_id": "cam0",
                "timestamp_epoch_s": 1.0,
                "liveness_pass": liveness,
                "liveness_confidence": 0.9,
                "quality_score": quality,
                "similarity_to_enroll": 0.8,
                "lighting": "indoor",
                "distance": "near",
                "glasses": "no",
                "metadata": {},
            }
        )
    )


def test_evaluate_dataset(tmp_path: Path) -> None:
    ds = tmp_path / "dataset"
    out = tmp_path / "artifacts"

    _write_session(ds, "alice", "s1", "bona_fide", np.array([1.0, 0.0, 0.0]), True, 0.9)
    _write_session(ds, "alice", "s2", "bona_fide", np.array([0.98, 0.01, 0.0]), True, 0.88)
    _write_session(ds, "bob", "s1", "bona_fide", np.array([0.0, 1.0, 0.0]), True, 0.87)
    _write_session(ds, "mallory", "s1", "attack_photo", np.array([0.95, 0.04, 0.0]), False, 0.3)

    artifact = evaluate_dataset(ds, out)
    assert "recommended" in artifact
    assert (out / "thresholds_dev.json").exists()
    assert (out / "report_dev.md").exists()
