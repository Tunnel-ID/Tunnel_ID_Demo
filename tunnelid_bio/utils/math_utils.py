from __future__ import annotations

from typing import Iterable, List

import numpy as np


EPS = 1e-12


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < EPS:
        return vec.copy()
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_n = l2_normalize(a)
    b_n = l2_normalize(b)
    return float(np.dot(a_n, b_n))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine_similarity(a, b))


def weighted_mean(embeddings: Iterable[np.ndarray], weights: Iterable[float]) -> np.ndarray:
    embeds = list(embeddings)
    w = np.asarray(list(weights), dtype=np.float64)
    stack = np.stack(embeds, axis=0).astype(np.float64)
    if len(embeds) == 0:
        raise ValueError("No embeddings provided")
    if w.shape[0] != stack.shape[0]:
        raise ValueError("Weights and embeddings length mismatch")
    w_sum = float(np.sum(w))
    if w_sum < EPS:
        w = np.ones_like(w)
        w_sum = float(np.sum(w))
    out = np.sum(stack * w[:, None], axis=0) / w_sum
    return out.astype(np.float32)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
