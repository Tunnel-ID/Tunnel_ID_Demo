import cv2
import numpy as np
from face_landmark import FaceLandmarkProcessor
from time import sleep
import json

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_vector(vector: np.ndarray, filename: str):
    with open(filename, "w") as f:
        json.dump(vector.tolist(), f)
    print(f"Saved vectpr to {filename}")

def capture_frame_with_preview(cap, countdown=3, label="Frame"):
    print(f"📸 Capturing {label} in {countdown} seconds...")

    for i in range(countdown, 0, -1):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("❌ Failed to read from webcam")
        cv2.putText(frame, f"{label} in {i}...", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.imshow("Camera", frame)
        cv2.waitKey(1000)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Failed to capture final frame")
    
    cv2.imshow(f"{label} Captured", frame)
    cv2.waitKey(1000)
    return frame

def main():
    processor = FaceLandmarkProcessor()
    cap = cv2.VideoCapture(0)

    frame1 = capture_frame_with_preview(cap, label="Frame 1")
    vec1 = processor.process(frame1)
    if vec1 is None:
        print("❌ No face detected in Frame 1")
        return
    save_vector(vec1, "vec1.json")


    frame2 = capture_frame_with_preview(cap, label="Frame 2")
    vec2 = processor.process(frame2)
    if vec2 is None:
        print("❌ No face detected in Frame 2")
        return
    save_vector(vec2, "vec2.json")

    cap.release()
    cv2.destroyAllWindows()

    # Compare the vectors
    cos_sim = cosine_similarity(vec1, vec2)
    l2_dist = np.linalg.norm(vec1 - vec2)

    print("\n✅ Comparison Results:")
    print(f"Cosine Similarity: {cos_sim:.6f}")
    print(f"L2 Distance:       {l2_dist:.6f}")

if __name__ == "__main__":
    main()
