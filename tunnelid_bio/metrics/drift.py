from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np

from tunnelid_bio.utils.math_utils import cosine_distance


@dataclass(frozen=True)
class DriftRecord:
    user_id: str
    session_id: str
    device_id: str
    embedding: np.ndarray


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": float(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
        "max": float(np.max(arr)),
    }


def compute_drift_stats(records: Sequence[DriftRecord]) -> Dict[str, object]:
    intra_session: List[float] = []
    inter_session: List[float] = []
    inter_device: List[float] = []

    by_user: Dict[str, List[DriftRecord]] = {}
    by_user_session: Dict[tuple[str, str], List[DriftRecord]] = {}
    for r in records:
        by_user.setdefault(r.user_id, []).append(r)
        by_user_session.setdefault((r.user_id, r.session_id), []).append(r)

    for items in by_user_session.values():
        for a, b in combinations(items, 2):
            d = cosine_distance(a.embedding, b.embedding)
            intra_session.append(d)

    for user_items in by_user.values():
        for a, b in combinations(user_items, 2):
            d = cosine_distance(a.embedding, b.embedding)
            if a.session_id != b.session_id:
                inter_session.append(d)
            if a.device_id != b.device_id:
                inter_device.append(d)

    return {
        "intra_session": _summary(intra_session),
        "inter_session": _summary(inter_session),
        "inter_device": _summary(inter_device),
    }


def recommend_drift_bound(
    distances: Sequence[float],
    epsilon_values: Sequence[float] = (0.01, 0.02, 0.05),
) -> Dict[str, float]:
    arr = np.asarray(list(distances), dtype=np.float64)
    if arr.size == 0:
        return {f"epsilon_{eps:g}": 0.0 for eps in epsilon_values}

    out: Dict[str, float] = {}
    for eps in epsilon_values:
        q = float(np.clip(1.0 - eps, 0.0, 1.0))
        out[f"epsilon_{eps:g}"] = float(np.quantile(arr, q))
    return out


def drift_risk(distance: float, drift_bound: float) -> bool:
    return bool(distance > drift_bound)
