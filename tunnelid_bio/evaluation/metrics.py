from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import roc_curve


def compute_roc(labels: np.ndarray, scores: np.ndarray) -> Dict[str, np.ndarray]:
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
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    thr = float(thresholds[idx])
    return eer, thr


def far_at_frr(
    fpr: np.ndarray,
    fnr: np.ndarray,
    thresholds: np.ndarray,
    frr_target: float,
) -> Tuple[float, float]:
    idx = int(np.nanargmin(np.abs(fnr - frr_target)))
    return float(fpr[idx]), float(thresholds[idx])
