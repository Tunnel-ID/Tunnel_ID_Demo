#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from tunnelid_bio.config import load_config
from tunnelid_bio.evaluation.harness import EvaluationHarness


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved biometric sessions")
    parser.add_argument("--sessions", required=True, help="Path to sessions npz")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    harness = EvaluationHarness.load_binary(cfg, args.sessions)
    report = harness.run()

    serializable = {
        k: v
        for k, v in report.items()
        if k not in {"genuine_scores", "impostor_scores", "roc"}
    }
    serializable["roc_points"] = len(report["roc"]["fpr"])

    print(json.dumps(serializable, indent=2, default=float))


if __name__ == "__main__":
    main()
