from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal, Optional, Sequence, Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from tunnelid_bio.metrics.roc_det import (
    bootstrap_metric_ci,
    compute_eer,
    far_at_fixed_frr,
    frr_at_fixed_far,
    roc_det_curve,
)

PolicyName = Literal["security-first", "ux-first", "balanced"]
CalibrationMethod = Literal["none", "platt", "isotonic"]


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    far: float
    frr: float
    policy: PolicyName


def _empirical_far_frr(impostor: np.ndarray, genuine: np.ndarray, threshold: float) -> Tuple[float, float]:
    far = float(np.mean(impostor >= threshold)) if impostor.size else 0.0
    frr = float(np.mean(genuine < threshold)) if genuine.size else 0.0
    return far, frr


def choose_threshold(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    policy: PolicyName,
    max_frr: float = 0.03,
    max_far: float = 1e-4,
) -> ThresholdChoice:
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    thresholds.sort()

    if thresholds.size == 0:
        return ThresholdChoice(threshold=1.0, far=1.0, frr=1.0, policy=policy)

    candidates: list[tuple[float, float, float]] = []
    for thr in thresholds:
        far, frr = _empirical_far_frr(impostor, genuine, float(thr))
        candidates.append((float(thr), far, frr))

    if policy == "balanced":
        labels = np.concatenate([np.ones_like(genuine, dtype=np.int32), np.zeros_like(impostor, dtype=np.int32)])
        scores = np.concatenate([genuine, impostor])
        roc = roc_det_curve(labels, scores)
        eer, thr = compute_eer(roc["fpr"], roc["fnr"], roc["thresholds"])
        return ThresholdChoice(threshold=thr, far=eer, frr=eer, policy=policy)

    if policy == "security-first":
        valid = [c for c in candidates if c[2] <= max_frr]
        if valid:
            thr, far, frr = min(valid, key=lambda c: (c[1], c[2], -c[0]))
        else:
            thr, far, frr = min(candidates, key=lambda c: (c[1], c[2], -c[0]))
        return ThresholdChoice(threshold=thr, far=far, frr=frr, policy=policy)

    # ux-first
    valid = [c for c in candidates if c[1] <= max_far]
    if valid:
        thr, far, frr = min(valid, key=lambda c: (c[2], c[1], c[0]))
    else:
        thr, far, frr = min(candidates, key=lambda c: (c[2], c[1], c[0]))
    return ThresholdChoice(threshold=thr, far=far, frr=frr, policy=policy)


def build_score_calibrator(
    labels: np.ndarray,
    scores: np.ndarray,
    method: CalibrationMethod = "none",
):
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    if method == "none":
        return None

    if method == "platt":
        model = LogisticRegression(random_state=2026, max_iter=1000)
        model.fit(scores.reshape(-1, 1), labels)
        return model

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, labels)
    return iso


def apply_calibrator(calibrator, scores: np.ndarray, method: CalibrationMethod) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if calibrator is None or method == "none":
        return scores
    if method == "platt":
        return np.asarray(calibrator.predict_proba(scores.reshape(-1, 1))[:, 1], dtype=np.float64)
    return np.asarray(calibrator.predict(scores), dtype=np.float64)


def calibration_report(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    policy: PolicyName,
    calibration_method: CalibrationMethod = "none",
    max_frr: float = 0.03,
    max_far: float = 1e-4,
    frr_targets: Sequence[float] = (0.01, 0.02, 0.03),
    far_targets: Sequence[float] = (1e-3, 1e-4, 1e-5),
    bootstrap_samples: int = 500,
) -> Dict[str, object]:
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)
    labels = np.concatenate([np.ones_like(genuine, dtype=np.int32), np.zeros_like(impostor, dtype=np.int32)])
    scores = np.concatenate([genuine, impostor])

    calibrator = build_score_calibrator(labels, scores, method=calibration_method)
    calibrated_scores = apply_calibrator(calibrator, scores, method=calibration_method)

    roc = roc_det_curve(labels, calibrated_scores)
    eer, eer_thr = compute_eer(roc["fpr"], roc["fnr"], roc["thresholds"])

    calibrated_genuine = calibrated_scores[: genuine.shape[0]]
    calibrated_impostor = calibrated_scores[genuine.shape[0] :]
    chosen = choose_threshold(
        calibrated_genuine,
        calibrated_impostor,
        policy=policy,
        max_frr=max_frr,
        max_far=max_far,
    )

    eer_ci = bootstrap_metric_ci(
        labels,
        calibrated_scores,
        metric_fn=lambda l, s: compute_eer(*[roc_det_curve(l, s)[k] for k in ("fpr", "fnr", "thresholds")])[0],
        n_bootstrap=bootstrap_samples,
    )

    return {
        "score_type": "cosine",
        "normalization": "raw" if calibration_method == "none" else calibration_method,
        "policy": policy,
        "threshold": {
            "tau_face": float(chosen.threshold),
            "far": float(chosen.far),
            "frr": float(chosen.frr),
        },
        "eer": float(eer),
        "eer_threshold": float(eer_thr),
        "eer_ci": {"low": eer_ci.low, "high": eer_ci.high},
        "far_at_fixed_frr": far_at_fixed_frr(roc["fpr"], roc["fnr"], roc["thresholds"], frr_targets),
        "frr_at_fixed_far": frr_at_fixed_far(roc["fpr"], roc["fnr"], roc["thresholds"], far_targets),
        "roc": {
            "fpr": roc["fpr"],
            "tpr": roc["tpr"],
            "fnr": roc["fnr"],
            "thresholds": roc["thresholds"],
        },
    }
