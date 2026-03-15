from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np


def pad_metrics(labels: Iterable[str], liveness_pass: Iterable[bool]) -> Dict[str, object]:
    rows = list(zip(labels, liveness_pass))
    attacks = [r for r in rows if r[0] != "bona_fide"]
    bona = [r for r in rows if r[0] == "bona_fide"]

    apcer = float(sum(1 for _, p in attacks if p) / len(attacks)) if attacks else 0.0
    bpcer = float(sum(1 for _, p in bona if not p) / len(bona)) if bona else 0.0
    acer = 0.5 * (apcer + bpcer)

    by_attack: Dict[str, Dict[str, float]] = {}
    attack_types = sorted({label for label, _ in attacks})
    for attack_type in attack_types:
        subset = [p for l, p in attacks if l == attack_type]
        by_attack[attack_type] = {
            "APCER": float(sum(1 for x in subset if x) / len(subset)) if subset else 0.0,
            "n": float(len(subset)),
        }

    return {
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": float(acer),
        "attack_types": by_attack,
        "n_attack": float(len(attacks)),
        "n_bona_fide": float(len(bona)),
    }


def pad_metrics_from_scores(
    labels: Sequence[str],
    scores: Sequence[float],
    threshold: float,
) -> Dict[str, object]:
    decisions = [float(s) >= float(threshold) for s in scores]
    out = pad_metrics(labels, decisions)
    out["threshold"] = float(threshold)
    out["score_summary"] = {
        "mean_bona_fide": float(np.mean([s for l, s in zip(labels, scores) if l == "bona_fide"]))
        if any(l == "bona_fide" for l in labels)
        else 0.0,
        "mean_attack": float(np.mean([s for l, s in zip(labels, scores) if l != "bona_fide"]))
        if any(l != "bona_fide" for l in labels)
        else 0.0,
    }
    return out
