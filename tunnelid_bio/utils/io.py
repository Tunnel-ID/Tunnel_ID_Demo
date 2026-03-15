from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_chmod(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except PermissionError:
        pass


def save_embedding_binary(path: str | Path, embedding: np.ndarray) -> None:
    p = Path(path)
    _safe_mkdir(p.parent)
    np.save(p, embedding.astype(np.float32))
    _safe_chmod(p)


def load_embedding_binary(path: str | Path) -> np.ndarray:
    return np.asarray(np.load(Path(path)), dtype=np.float32)


def save_json(path: str | Path, data: Dict[str, Any]) -> None:
    p = Path(path)
    _safe_mkdir(p.parent)
    p.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    _safe_chmod(p)


def load_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_template_set(path: str | Path, embeddings: np.ndarray) -> None:
    p = Path(path)
    _safe_mkdir(p.parent)
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    np.savez_compressed(p, embeddings=arr)
    _safe_chmod(p)


def load_template_set(path: str | Path) -> np.ndarray:
    data = np.load(Path(path), allow_pickle=False)
    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr
    arr = np.asarray(data["embeddings"], dtype=np.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def resolve_template_paths(template_dir: str | Path, user_id: str) -> Tuple[Path, Path, Path]:
    tdir = Path(template_dir)
    return (
        tdir / f"{user_id}.npz",
        tdir / f"{user_id}.npy",
        tdir / f"{user_id}.meta.json",
    )
