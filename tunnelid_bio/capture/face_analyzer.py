from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tunnelid_bio.types import FaceObservation

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


@dataclass
class FaceAnalyzer:
    min_detection_confidence: float = 0.85

    def __post_init__(self) -> None:
        if mp is None:
            raise RuntimeError(
                "mediapipe is required for capture/landmark analysis. Install requirements first."
            )
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._mesh.close()

    @staticmethod
    def _pose_from_geometry(points: np.ndarray) -> tuple[float, float, float]:
        # Stable pseudo-pose from eye-nose-mouth geometry.
        # Sign convention: yaw_left is negative, yaw_right is positive.
        right_eye = points[0]
        left_eye = points[1]
        nose = points[2]
        mouth = points[3]

        eye_mid = 0.5 * (left_eye + right_eye)
        eye_dist = float(np.linalg.norm(left_eye - right_eye) + 1e-6)

        yaw_norm = float((nose[0] - eye_mid[0]) / (0.5 * eye_dist))
        yaw_deg = float(np.clip(yaw_norm * 35.0, -45.0, 45.0))

        mouth_dist = float(abs(mouth[1] - eye_mid[1]) + 1e-6)
        pitch_norm = float((nose[1] - eye_mid[1]) / mouth_dist - 0.50)
        pitch_deg = float(np.clip(pitch_norm * 60.0, -35.0, 35.0))

        dy = float(left_eye[1] - right_eye[1])
        dx = float(left_eye[0] - right_eye[0] + 1e-6)
        roll_deg = float(np.degrees(np.arctan2(dy, dx)))
        return yaw_deg, pitch_deg, roll_deg

    def detect(self, image_bgr: np.ndarray, frame_id: int) -> Optional[FaceObservation]:
        image_rgb = image_bgr[:, :, ::-1]
        result = self._mesh.process(image_rgb)
        if not result.multi_face_landmarks:
            return None

        h, w = image_bgr.shape[:2]
        lm = result.multi_face_landmarks[0].landmark
        all_pts = np.asarray([(p.x * w, p.y * h) for p in lm], dtype=np.float32)
        x1 = int(np.min(all_pts[:, 0]))
        y1 = int(np.min(all_pts[:, 1]))
        x2 = int(np.max(all_pts[:, 0]))
        y2 = int(np.max(all_pts[:, 1]))

        # 6 points for compatibility with existing downstream code.
        # [right_eye, left_eye, nose, mouth_center, right_cheek, left_cheek]
        right_eye = np.mean(all_pts[[33, 133, 159, 145]], axis=0)
        left_eye = np.mean(all_pts[[362, 263, 386, 374]], axis=0)
        nose = all_pts[1]
        mouth_center = 0.5 * (all_pts[13] + all_pts[14])
        right_cheek = all_pts[234]
        left_cheek = all_pts[454]
        landmarks = np.stack(
            [right_eye, left_eye, nose, mouth_center, right_cheek, left_cheek], axis=0
        ).astype(np.float32)

        yaw, pitch, roll = self._pose_from_geometry(landmarks)

        return FaceObservation(
            frame_id=frame_id,
            bbox_xyxy=(x1, y1, x2, y2),
            landmarks=landmarks,
            detection_confidence=1.0,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )
