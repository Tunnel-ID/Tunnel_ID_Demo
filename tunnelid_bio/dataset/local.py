from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np

from tunnelid_bio.metrics.drift import DriftRecord


VALID_LABELS = {
    "bona_fide",
    "attack_print",
    "attack_photo",
    "attack_screen",
    "attack_video",
    "attack_deepfake",
    "attack_deepfake_screen",
}


@dataclass(frozen=True)
class DatasetSession:
    user_id: str
    session_id: str
    label: str
    device_id: str
    embedding: np.ndarray
    liveness_pass: bool
    liveness_confidence: float
    quality_score: float
    similarity_to_enroll: float
    timestamp_epoch_s: float
    lighting: str
    distance: str
    glasses: str
    metadata: Dict[str, object]


def _safe_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.chmod(path, 0o600)


def _safe_write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(arr, dtype=np.float32))
    os.chmod(path, 0o600)


def _session_dir(dataset_root: Path, user_id: str, session_id: str) -> Path:
    return dataset_root / "users" / user_id / session_id


def store_session(
    dataset_root: str | Path,
    user_id: str,
    label: str,
    device_id: str,
    embedding: np.ndarray,
    liveness_pass: bool,
    liveness_confidence: float,
    quality_score: float,
    similarity_to_enroll: float,
    metadata: Dict[str, object],
    lighting: str = "unknown",
    distance: str = "unknown",
    glasses: str = "unknown",
    store_frames: bool = False,
    frames_bgr: Optional[Iterable[np.ndarray]] = None,
) -> Path:
    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {label}")

    now = time.time()
    session_id = f"session_{int(now)}_{device_id}"
    out_dir = _session_dir(Path(dataset_root), user_id, session_id)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if store_frames and frames_bgr is not None:
        for i, frame in enumerate(frames_bgr):
            cv2.imwrite(str(frames_dir / f"{i:04d}.jpg"), frame)

    _safe_write_npy(out_dir / "template.npy", np.asarray(embedding, dtype=np.float32))

    meta = {
        "user_id": user_id,
        "session_id": session_id,
        "label": label,
        "device_id": device_id,
        "timestamp_epoch_s": now,
        "liveness_pass": bool(liveness_pass),
        "liveness_confidence": float(liveness_confidence),
        "quality_score": float(quality_score),
        "similarity_to_enroll": float(similarity_to_enroll),
        "lighting": lighting,
        "distance": distance,
        "glasses": glasses,
        "store_frames": bool(store_frames),
        "metadata": metadata,
    }
    _safe_write_json(out_dir / "meta.json", meta)
    return out_dir


def load_dataset_sessions(dataset_root: str | Path) -> List[DatasetSession]:
    root = Path(dataset_root) / "users"
    if not root.exists():
        return []

    sessions: List[DatasetSession] = []
    for meta_path in sorted(root.glob("*/*/meta.json")):
        base_dir = meta_path.parent
        emb_path = base_dir / "template.npy"
        if not emb_path.exists():
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        emb = np.asarray(np.load(emb_path), dtype=np.float32)

        sessions.append(
            DatasetSession(
                user_id=str(meta.get("user_id", "")),
                session_id=str(meta.get("session_id", base_dir.name)),
                label=str(meta.get("label", "bona_fide")),
                device_id=str(meta.get("device_id", "unknown")),
                embedding=emb,
                liveness_pass=bool(meta.get("liveness_pass", False)),
                liveness_confidence=float(meta.get("liveness_confidence", 0.0)),
                quality_score=float(meta.get("quality_score", 0.0)),
                similarity_to_enroll=float(meta.get("similarity_to_enroll", 0.0)),
                timestamp_epoch_s=float(meta.get("timestamp_epoch_s", 0.0)),
                lighting=str(meta.get("lighting", "unknown")),
                distance=str(meta.get("distance", "unknown")),
                glasses=str(meta.get("glasses", "unknown")),
                metadata=dict(meta.get("metadata", {})),
            )
        )
    return sessions


def drift_records_from_sessions(sessions: Iterable[DatasetSession]) -> List[DriftRecord]:
    return [
        DriftRecord(
            user_id=s.user_id,
            session_id=s.session_id,
            device_id=s.device_id,
            embedding=s.embedding,
        )
        for s in sessions
        if s.label == "bona_fide"
    ]
