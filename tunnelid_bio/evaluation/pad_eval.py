from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np

from tunnelid_bio.dataset.local import load_dataset_sessions
from tunnelid_bio.metrics.pad_iso import pad_metrics_from_scores
from tunnelid_bio.metrics.roc_det import compute_eer, roc_det_curve


def _normalized_label(label: str) -> str:
    if label in {"bona_fide", "bonafide", "real"}:
        return "bona_fide"
    if label in {"attack_photo", "attack_print", "print_photo"}:
        return "attack_print"
    if label in {"attack_screen", "screen_replay"}:
        return "attack_screen"
    if label in {"attack_video", "video_replay"}:
        return "attack_video"
    if label in {"attack_deepfake", "attack_deepfake_screen", "deepfake_screen"}:
        return "attack_deepfake_screen"
    return label


def _extract_pad_score(session) -> float:
    meta = session.metadata or {}
    out_meta = meta.get("output_metadata", {}) if isinstance(meta, dict) else {}
    if isinstance(out_meta, dict):
        lcomp = out_meta.get("liveness_components", {})
        if isinstance(lcomp, dict) and "overall_confidence" in lcomp:
            return float(lcomp["overall_confidence"])
    return float(session.liveness_confidence)


def _find_threshold(scores: np.ndarray, labels01: np.ndarray, target_bpcer: float = 0.03) -> Dict[str, float]:
    thresholds = np.unique(scores)
    if thresholds.size == 0:
        return {"threshold": 0.5, "apcer": 1.0, "bpcer": 1.0}

    best = None
    for thr in thresholds:
        bona = scores[labels01 == 1]
        attack = scores[labels01 == 0]
        bpcer = float(np.mean(bona < thr)) if bona.size else 1.0
        apcer = float(np.mean(attack >= thr)) if attack.size else 1.0
        row = (apcer, bpcer, float(thr))
        if bpcer <= target_bpcer:
            if best is None or row[0] < best[0] or (row[0] == best[0] and row[2] > best[2]):
                best = row

    if best is None:
        # fallback balanced EER point
        roc = roc_det_curve(labels01, scores)
        eer, thr = compute_eer(roc["fpr"], roc["fnr"], roc["thresholds"])
        return {"threshold": float(thr), "apcer": float(eer), "bpcer": float(eer)}

    return {"threshold": best[2], "apcer": best[0], "bpcer": best[1]}


def _write_hist(path: Path, bona_scores: np.ndarray, attack_scores: np.ndarray) -> None:
    w, h = 980, 640
    margin = 70
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (0, 0, 0), 1)
    cv2.putText(img, "PAD score distribution", (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

    all_vals = np.concatenate([bona_scores, attack_scores]) if bona_scores.size and attack_scores.size else np.concatenate([bona_scores, attack_scores, np.array([0.0, 1.0])])
    lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
    span = max(1e-6, hi - lo)

    def draw_hist(vals: np.ndarray, color: tuple[int, int, int]) -> None:
        if vals.size == 0:
            return
        hist, edges = np.histogram(vals, bins=25, range=(lo, hi))
        hist = hist.astype(np.float64)
        hist /= max(1.0, np.max(hist))
        for i, v in enumerate(hist):
            x1 = int(margin + (edges[i] - lo) / span * (w - 2 * margin))
            x2 = int(margin + (edges[i + 1] - lo) / span * (w - 2 * margin)) - 2
            y1 = int(h - margin - v * (h - 2 * margin))
            y2 = h - margin
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

    draw_hist(bona_scores, (30, 140, 40))
    draw_hist(attack_scores, (200, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def run_pad_eval(dataset_root: str | Path, out_dir: str | Path, target_bpcer: float = 0.03) -> Dict[str, object]:
    sessions = load_dataset_sessions(dataset_root)
    if not sessions:
        raise ValueError("No sessions found in dataset")

    labels = [_normalized_label(s.label) for s in sessions]
    scores = np.asarray([_extract_pad_score(s) for s in sessions], dtype=np.float64)
    labels01 = np.asarray([1 if l == "bona_fide" else 0 for l in labels], dtype=np.int32)

    if np.unique(labels01).shape[0] < 2:
        raise ValueError("PAD eval requires both bona_fide and attack samples")

    selected = _find_threshold(scores, labels01, target_bpcer=target_bpcer)
    metrics = pad_metrics_from_scores(labels, scores.tolist(), threshold=selected["threshold"])
    metrics["selected_threshold"] = selected

    roc = roc_det_curve(labels01, scores)
    eer, eer_thr = compute_eer(roc["fpr"], roc["fnr"], roc["thresholds"])
    metrics["roc"] = {
        "fpr": roc["fpr"],
        "tpr": roc["tpr"],
        "fnr": roc["fnr"],
        "thresholds": roc["thresholds"],
    }
    metrics["eer"] = float(eer)
    metrics["eer_threshold"] = float(eer_thr)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bona_scores = scores[labels01 == 1]
    attack_scores = scores[labels01 == 0]
    _write_hist(out / "pad_score_hist.png", bona_scores, attack_scores)

    artifact = {
        "version": "pad_thresholds@v1",
        "threshold": float(selected["threshold"]),
        "target_bpcer": float(target_bpcer),
        "metrics": {
            k: v for k, v in metrics.items() if k != "roc"
        },
    }

    (out / "pad_thresholds_dev.json").write_text(json.dumps(artifact, indent=2, default=float), encoding="utf-8")
    (out / "pad_report_dev.md").write_text(
        "\n".join(
            [
                "# PAD Evaluation Report",
                "",
                f"- n_samples: {len(labels)}",
                f"- threshold: {selected['threshold']:.6f}",
                f"- APCER: {metrics['APCER']:.6f}",
                f"- BPCER: {metrics['BPCER']:.6f}",
                f"- ACER: {metrics['ACER']:.6f}",
                f"- EER: {eer:.6f}",
                "",
                "## Labels",
                "- bona_fide",
                "- attack_print",
                "- attack_screen",
                "- attack_video",
                "- attack_deepfake_screen",
            ]
        ),
        encoding="utf-8",
    )
    return artifact
