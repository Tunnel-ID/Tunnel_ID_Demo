from __future__ import annotations

import numpy as np

from tunnelid_bio.calibration.thresholds import choose_threshold
from tunnelid_bio.metrics.drift import recommend_drift_bound
from tunnelid_bio.metrics.pad_iso import pad_metrics
from tunnelid_bio.metrics.roc_det import compute_eer, roc_det_curve


def test_roc_and_eer_runs() -> None:
    labels = np.array([1] * 20 + [0] * 20, dtype=np.int32)
    scores = np.array([0.8] * 20 + [0.2] * 20, dtype=np.float64)
    roc = roc_det_curve(labels, scores)
    eer, thr = compute_eer(roc["fpr"], roc["fnr"], roc["thresholds"])
    assert 0.0 <= eer <= 0.5
    assert -1.0 <= thr <= 1.0


def test_threshold_policies() -> None:
    genuine = np.array([0.7, 0.8, 0.9, 0.75], dtype=np.float64)
    impostor = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)

    sec = choose_threshold(genuine, impostor, policy="security-first", max_frr=0.25)
    ux = choose_threshold(genuine, impostor, policy="ux-first", max_far=0.01)
    bal = choose_threshold(genuine, impostor, policy="balanced")

    assert -1.0 <= sec.threshold <= 1.0
    assert -1.0 <= ux.threshold <= 1.0
    assert -1.0 <= bal.threshold <= 1.0


def test_pad_metrics() -> None:
    labels = ["bona_fide", "bona_fide", "attack_photo", "attack_screen"]
    liveness = [True, False, False, True]
    out = pad_metrics(labels, liveness)

    assert "APCER" in out and "BPCER" in out and "ACER" in out
    assert 0.0 <= out["APCER"] <= 1.0
    assert 0.0 <= out["BPCER"] <= 1.0


def test_recommend_drift_bound_monotonic() -> None:
    d = [0.1, 0.2, 0.25, 0.3, 0.4]
    out = recommend_drift_bound(d, epsilon_values=(0.05, 0.02, 0.01))
    assert out["epsilon_0.05"] <= out["epsilon_0.02"] <= out["epsilon_0.01"]
