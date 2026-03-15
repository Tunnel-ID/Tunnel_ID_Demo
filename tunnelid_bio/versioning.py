from __future__ import annotations

from typing import Dict

from tunnelid_bio.config import PipelineConfig


def get_version_tags(cfg: PipelineConfig) -> Dict[str, str]:
    return {
        "encoder": cfg.versions.encoder,
        "preprocessing": cfg.versions.preprocessing,
        "liveness": cfg.versions.liveness,
        "aggregation": cfg.versions.aggregation,
    }
