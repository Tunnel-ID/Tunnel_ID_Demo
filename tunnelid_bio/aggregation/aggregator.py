from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from tunnelid_bio.config import AggregationConfig
from tunnelid_bio.types import AggregationDiagnostics
from tunnelid_bio.utils.math_utils import cosine_distance, l2_normalize, weighted_mean


@dataclass
class MultiFrameAggregator:
    cfg: AggregationConfig

    def aggregate(
        self,
        embeddings: Sequence[np.ndarray],
        quality_weights: Sequence[float],
    ) -> Tuple[np.ndarray, AggregationDiagnostics]:
        if len(embeddings) < self.cfg.min_frames_total:
            raise ValueError(f"Need at least {self.cfg.min_frames_total} embeddings")

        base_mean = l2_normalize(weighted_mean(embeddings, quality_weights))
        distances = [cosine_distance(e, base_mean) for e in embeddings]

        keep_idx = [
            i
            for i, d in enumerate(distances)
            if d <= self.cfg.max_outlier_cosine_distance
        ]
        if len(keep_idx) < self.cfg.min_frames_total:
            raise ValueError("Too many outliers during aggregation")

        kept_embeddings = [embeddings[i] for i in keep_idx]
        kept_weights = [quality_weights[i] for i in keep_idx]
        final = l2_normalize(weighted_mean(kept_embeddings, kept_weights))

        kept_distances = [cosine_distance(e, final) for e in kept_embeddings]
        variance = float(np.var(np.asarray(kept_distances, dtype=np.float64)))

        diag = AggregationDiagnostics(
            used_count=len(keep_idx),
            dropped_count=len(embeddings) - len(keep_idx),
            per_frame_cosine_distance=kept_distances,
            intra_session_variance=variance,
        )
        return final.astype(np.float32), diag
