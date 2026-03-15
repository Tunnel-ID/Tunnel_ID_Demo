#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from tunnelid_bio.config import load_config
from tunnelid_bio.evaluation.harness import EvaluationHarness


def search_threshold(genuine: np.ndarray, impostor: np.ndarray, target_far: float) -> dict:
    grid = np.linspace(-1.0, 1.0, 2001)
    best_thr = 1.0
    best_frr = 1.0

    for thr in grid:
        far = float(np.mean(impostor >= thr))
        frr = float(np.mean(genuine < thr))
        if far <= target_far and frr < best_frr:
            best_frr = frr
            best_thr = float(thr)

    if best_thr == 1.0:
        # No threshold satisfies target FAR; fallback to minimum FAR point.
        pairs = []
        for thr in grid:
            far = float(np.mean(impostor >= thr))
            frr = float(np.mean(genuine < thr))
            pairs.append((far, frr, float(thr)))
        pairs.sort(key=lambda x: (x[0], x[1]))
        far, frr, best_thr = pairs[0]
        return {
            "threshold": best_thr,
            "FAR": far,
            "FRR": frr,
            "note": "target_far_unreachable_fallback_min_far",
        }

    far = float(np.mean(impostor >= best_thr))
    frr = float(np.mean(genuine < best_thr))
    return {
        "threshold": best_thr,
        "FAR": far,
        "FRR": frr,
        "note": "target_far_satisfied",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate similarity threshold from evaluation sessions")
    parser.add_argument("--sessions", required=True, help="Path to evaluation sessions npz")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--target-far", type=float, default=0.001)
    parser.add_argument("--output", default="configs/calibrated.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    harness = EvaluationHarness.load_binary(cfg, args.sessions)
    report = harness.run()

    genuine = report["genuine_scores"]
    impostor = report["impostor_scores"]
    calibration = search_threshold(genuine, impostor, target_far=args.target_far)

    out = {
        "embedding": {
            "min_cosine_similarity_match": float(calibration["threshold"]),
        },
        "calibration_report": {
            "target_far": args.target_far,
            "result_far": calibration["FAR"],
            "result_frr": calibration["FRR"],
            "eer": report["EER"],
            "eer_threshold": report["EER_threshold"],
            "far_at_frr_target": report["FAR_at_FRR_target"],
            "far_threshold": report["FAR_threshold"],
            "note": calibration["note"],
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
