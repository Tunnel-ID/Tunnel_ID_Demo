from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from tunnelid_bio.calibration.thresholds import calibration_report, choose_threshold
from tunnelid_bio.dataset.local import DatasetSession, drift_records_from_sessions, load_dataset_sessions
from tunnelid_bio.metrics.drift import compute_drift_stats, recommend_drift_bound
from tunnelid_bio.metrics.pad_iso import pad_metrics
from tunnelid_bio.metrics.roc_det import score_summary
from tunnelid_bio.utils.math_utils import cosine_distance, cosine_similarity


def _pairwise_scores(sessions: Sequence[DatasetSession]) -> Tuple[np.ndarray, np.ndarray]:
    bona = [s for s in sessions if s.label == "bona_fide"]
    genuine: List[float] = []
    impostor: List[float] = []

    for a, b in combinations(bona, 2):
        score = cosine_similarity(a.embedding, b.embedding)
        if a.user_id == b.user_id:
            genuine.append(score)
        else:
            impostor.append(score)

    return np.asarray(genuine, dtype=np.float64), np.asarray(impostor, dtype=np.float64)


def _inter_session_dists(sessions: Sequence[DatasetSession]) -> List[float]:
    bona = [s for s in sessions if s.label == "bona_fide"]
    out: List[float] = []
    for a, b in combinations(bona, 2):
        if a.user_id == b.user_id and a.session_id != b.session_id:
            out.append(cosine_distance(a.embedding, b.embedding))
    return out


def _quality_gate_threshold(sessions: Sequence[DatasetSession], drift_bound: float) -> float:
    bona = [s for s in sessions if s.label == "bona_fide"]
    if not bona:
        return 0.0

    # Conservative heuristic: use lower percentile quality among sessions with acceptable drift proxy.
    accepted_quality = [s.quality_score for s in bona if (1.0 - s.similarity_to_enroll) <= drift_bound]
    if not accepted_quality:
        accepted_quality = [s.quality_score for s in bona]
    return float(np.quantile(np.asarray(accepted_quality, dtype=np.float64), 0.10))


def _write_simple_plot(path: Path, x: np.ndarray, y: np.ndarray, title: str, xlab: str, ylab: str) -> None:
    w, h = 900, 620
    margin = 70
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (0, 0, 0), 1)
    cv2.putText(img, title, (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, xlab, (w // 2 - 40, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, ylab, (8, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    if x.size and y.size:
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        x_span = max(1e-8, x_max - x_min)
        y_span = max(1e-8, y_max - y_min)

        pts = []
        for xv, yv in zip(x, y):
            px = int(margin + (xv - x_min) / x_span * (w - 2 * margin))
            py = int(h - margin - (yv - y_min) / y_span * (h - 2 * margin))
            pts.append((px, py))

        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], (30, 90, 220), 2, cv2.LINE_AA)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def _write_hist(path: Path, values: np.ndarray, title: str, xlab: str) -> None:
    w, h = 900, 620
    margin = 70
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (0, 0, 0), 1)
    cv2.putText(img, title, (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, xlab, (w // 2 - 50, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    if values.size:
        hist, _ = np.histogram(values, bins=20)
        hist = hist.astype(np.float64)
        hist /= max(1.0, np.max(hist))
        bar_w = (w - 2 * margin) / 20.0
        for i, v in enumerate(hist):
            x1 = int(margin + i * bar_w)
            x2 = int(margin + (i + 1) * bar_w - 2)
            y1 = int(h - margin - v * (h - 2 * margin))
            y2 = h - margin
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 120, 230), -1)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def evaluate_dataset(
    dataset_root: str | Path,
    out_dir: str | Path,
    policy: str = "security-first",
    calibration_method: str = "none",
    max_frr: float = 0.03,
    max_far: float = 1e-4,
) -> Dict[str, object]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sessions = load_dataset_sessions(dataset_root)
    genuine, impostor = _pairwise_scores(sessions)
    if genuine.size == 0 or impostor.size == 0:
        raise ValueError("Not enough bona fide sessions for genuine/impostor evaluation")

    cal = calibration_report(
        genuine,
        impostor,
        policy=policy,  # type: ignore[arg-type]
        calibration_method=calibration_method,  # type: ignore[arg-type]
        max_frr=max_frr,
        max_far=max_far,
    )

    pad = pad_metrics([s.label for s in sessions], [s.liveness_pass for s in sessions])
    drift_stats = compute_drift_stats(drift_records_from_sessions(sessions))
    inter_session = _inter_session_dists(sessions)
    delta_map = recommend_drift_bound(inter_session, epsilon_values=(0.01, 0.02, 0.05))

    drift_bound = float(delta_map.get("epsilon_0.02", 0.0))
    quality_thr = _quality_gate_threshold(sessions, drift_bound=drift_bound)

    rec = {
        "tau_face": float(cal["threshold"]["tau_face"]),
        "tau_face_policy": policy,
        "score_type": cal["score_type"],
        "score_normalization": cal["normalization"],
        "drift_bound_delta": drift_bound,
        "drift_bound_by_epsilon": delta_map,
        "quality_gate_threshold": quality_thr,
        "max_far_target": max_far,
        "max_frr_target": max_frr,
        "far_at_tau_face": float(cal["threshold"]["far"]),
        "frr_at_tau_face": float(cal["threshold"]["frr"]),
        "pad": pad,
        "drift": drift_stats,
        "genuine_summary": score_summary(genuine),
        "impostor_summary": score_summary(impostor),
    }

    thresholds_artifact = {
        "version": "thresholds@v1",
        "recommended": rec,
        "calibration": {
            k: v
            for k, v in cal.items()
            if k != "roc"
        },
    }

    (out_path / "thresholds_dev.json").write_text(json.dumps(thresholds_artifact, indent=2, default=float))

    roc = cal["roc"]
    _write_simple_plot(out_path / "roc_curve.png", np.asarray(roc["fpr"]), np.asarray(roc["tpr"]), "ROC", "FAR", "TAR")
    _write_simple_plot(out_path / "det_curve.png", np.asarray(roc["fpr"]), np.asarray(roc["fnr"]), "DET", "FAR", "FRR")
    _write_hist(out_path / "genuine_hist.png", genuine, "Genuine Score Histogram", "Cosine Similarity")
    _write_hist(out_path / "impostor_hist.png", impostor, "Impostor Score Histogram", "Cosine Similarity")
    _write_hist(out_path / "drift_hist.png", np.asarray(inter_session, dtype=np.float64), "Inter-session Drift Histogram", "Cosine Distance")

    report_md = [
        "# TunnelID Evaluation Report",
        "",
        f"- Sessions: {len(sessions)}",
        f"- Policy: {policy}",
        f"- Recommended tau_face: {rec['tau_face']:.6f}",
        f"- FAR@tau: {rec['far_at_tau_face']:.6f}",
        f"- FRR@tau: {rec['frr_at_tau_face']:.6f}",
        f"- Recommended drift bound delta (epsilon=0.02): {rec['drift_bound_delta']:.6f}",
        f"- Quality gate threshold: {rec['quality_gate_threshold']:.6f}",
        "",
        "## PAD",
        f"- APCER: {pad['APCER']:.6f}",
        f"- BPCER: {pad['BPCER']:.6f}",
        f"- ACER: {pad['ACER']:.6f}",
        "",
        "## Plots",
        "- roc_curve.png",
        "- det_curve.png",
        "- genuine_hist.png",
        "- impostor_hist.png",
        "- drift_hist.png",
    ]
    (out_path / "report_dev.md").write_text("\n".join(report_md), encoding="utf-8")

    return thresholds_artifact
