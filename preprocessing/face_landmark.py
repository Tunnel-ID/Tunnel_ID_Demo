import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional

class FaceLandmarkProcessor:
    def __init__(self, image_size: Tuple[int, int]=(224, 224)):
        self.image_size = image_size
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode = True,
                                           max_num_faces = 1,
                                           refine_landmarks = True,
                                           min_detection_confidence=0.5)
    
    def preprocess_lighting(self, image: np.ndarray) -> np.ndarray:
        """ Apply CLAHE to normalize lightning"""
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    def extract_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detetct facial landmarks"""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array([
                [pt.x * image.shape[1], pt.y * image.shape[0]]
                for pt in face_landmarks.landmark
            ], dtype = np.float64)
            return landmarks
        return None
    
    def mask_background(self, image: np.ndarray, landmarks: np.ndarray)-> np.ndarray:
        """ Apply convex hull mask to remove background"""
        hull = cv2.convexHull(landmarks.astype(np.int32))
        mask = np.zeros_like(image)
        cv2.fillConvexPoly(mask, hull, (255, 255, 255))
        return cv2.bitwise_and(image, mask)
    
    def align_and_crop(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """ Crop to bounding box of landmarks and resize"""
        x_min, y_min = np.min(landmarks, axis=0).astype(int)
        x_max, y_max = np.max(landmarks, axis=0).astype(int)
        cropped = image[y_min: y_max, x_min:x_max]
        return cv2.resize(cropped, self.image_size)
    
    def vectorize_landmarks(self, landmarks:np.ndarray) -> np.ndarray:
        """Convert landmark coordinates into flattened normalized vector"""
        vec = landmarks.flatten().astype(np.float64)
        norm = np.linalg.norm(vec)
        return vec/norm if norm!=0 else vec
    
    def process(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Main pipeline: preprocess, extract landmarks, mask, align, vectorize"""
        image_eq = self.preprocess_lighting(image)
        landmarks = self. extract_landmarks(image_eq)
        if landmarks is None:
            return None
        masked_face = self.mask_background(image_eq, landmarks)
        aligned_face = self.align_and_crop(masked_face, landmarks)
        vector  = self.vectorize_landmarks(landmarks)
        return vector    