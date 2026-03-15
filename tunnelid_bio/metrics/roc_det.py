from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_curve


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float


def roc_det_curve(labels: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    return {
        "fpr": fpr,
        "tpr": tpr,
        "fnr": fnr,
        "thresholds": thresholds,
    }


def compute_eer(fpr: np.ndarray, fnr: np.ndarray, thresholds: np.ndarray) -> Tuple[float, float]:
    idx = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[idx] + fnr[idx]) * 0.5), float(thresholds[idx])


def far_at_fixed_frr(
    fpr: np.ndarray,
    fnr: np.ndarray,
    thresholds: np.ndarray,
    frr_targets: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for target in frr_targets:
        idx = int(np.nanargmin(np.abs(fnr - target)))
        out[f"FRR_{target:g}"] = {
            "frr": float(fnr[idx]),
            "far": float(fpr[idx]),
            "threshold": float(thresholds[idx]),
        }
    return out


def frr_at_fixed_far(
    fpr: np.ndarray,
    fnr: np.ndarray,
    thresholds: np.ndarray,
    far_targets: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for target in far_targets:
        idx = int(np.nanargmin(np.abs(fpr - target)))
        out[f"FAR_{target:g}"] = {
            "far": float(fpr[idx]),
            "frr": float(fnr[idx]),
            "threshold": float(thresholds[idx]),
        }
    return out


def _bootstrap_sample(labels: np.ndarray, scores: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    n = labels.shape[0]
    idx = rng.integers(0, n, size=n)
    return labels[idx], scores[idx]


def bootstrap_metric_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.95,
    seed: int = 2026,
) -> ConfidenceInterval:
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    rng = np.random.default_rng(seed)

    vals: List[float] = []
    for _ in range(n_bootstrap):
        b_labels, b_scores = _bootstrap_sample(labels, scores, rng)
        if np.unique(b_labels).shape[0] < 2:
            continue
        vals.append(float(metric_fn(b_labels, b_scores)))

    if not vals:
        return ConfidenceInterval(low=0.0, high=0.0)

    lo_q = (1.0 - ci_alpha) * 0.5
    hi_q = 1.0 - lo_q
    arr = np.asarray(vals, dtype=np.float64)
    return ConfidenceInterval(
        low=float(np.quantile(arr, lo_q)),
        high=float(np.quantile(arr, hi_q)),
    )


def score_summary(scores: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(scores), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0.0, "mean": 0.0, "std": 0.0, "p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
    }
