from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

from tunnelid_bio.config import load_config
from tunnelid_bio.pipeline.service import BiometricPipeline
from tunnelid_bio.types import FramePacket


@lru_cache(maxsize=2)
def _default_pipeline(config_path: str | None = None) -> BiometricPipeline:
    return BiometricPipeline(load_config(config_path))


def enroll_session(
    frames_sequence: Sequence[np.ndarray | FramePacket],
    config_path: str | None = None,
) -> dict:
    return _default_pipeline(config_path).enroll_session(frames_sequence)


def verify_session(
    frames_sequence: Sequence[np.ndarray | FramePacket],
    stored_embedding: np.ndarray,
    config_path: str | None = None,
) -> dict:
    return _default_pipeline(config_path).verify_session(frames_sequence, stored_embedding)
