from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from tunnelid_bio.config import EmbeddingConfig
from tunnelid_bio.types import FaceObservation
from tunnelid_bio.utils.math_utils import l2_normalize

try:
    from insightface.app import FaceAnalysis
except ImportError:  # pragma: no cover
    FaceAnalysis = None


@dataclass
class ArcFaceEncoder:
    cfg: EmbeddingConfig

    def __post_init__(self) -> None:
        if FaceAnalysis is None:
            raise RuntimeError(
                "insightface is required for ArcFace embeddings. Install requirements first."
            )
        self._app = FaceAnalysis(name=self.cfg.model_name, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(self.cfg.det_size, self.cfg.det_size))

    @staticmethod
    def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = float(iw * ih)
        if inter <= 0:
            return 0.0
        a_area = float(max(1, ax2 - ax1) * max(1, ay2 - ay1))
        b_area = float(max(1, bx2 - bx1) * max(1, by2 - by1))
        return inter / (a_area + b_area - inter + 1e-12)

    def _select_face(self, faces: list, hint: Optional[FaceObservation]) -> Optional[object]:
        if not faces:
            return None
        if hint is None:
            return max(faces, key=lambda f: float(f.det_score))

        hint_box = hint.bbox_xyxy
        scored = []
        for f in faces:
            box = tuple(int(v) for v in f.bbox)
            iou = self._bbox_iou((box[0], box[1], box[2], box[3]), hint_box)
            scored.append((iou, float(f.det_score), f))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return scored[0][2]

    def encode(self, image_bgr: np.ndarray, face_hint: Optional[FaceObservation] = None) -> np.ndarray:
        faces = self._app.get(image_bgr)
        face = self._select_face(faces, face_hint)
        if face is None:
            raise ValueError("No face found for embedding extraction")

        emb = np.asarray(face.embedding, dtype=np.float32)
        if emb.shape[0] != self.cfg.expected_dim:
            raise ValueError(
                f"Unexpected embedding dim {emb.shape[0]}; expected {self.cfg.expected_dim}"
            )
        return l2_normalize(emb)
