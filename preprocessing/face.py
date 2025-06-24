import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from insightface.app import FaceAnalysis
import logging

# Initialize ArcFace model

def initialize_face_analysis():
    try:
        # Try GPU first
        logging.info("Trying to initialize FaceAnalysis with GPU...")
        app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        logging.info("GPU face analysis initialized")
    except Exception as e:
        logging.warning(f"GPU unavailable: {e}")
        logging.info(" Falling back to CPU..")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        logging.info("✅ CPU FaceAnalysis initialized")
    return app

# Initialize globally
app = initialize_face_analysis()


# preprocessing/face.py

def decode_and_align_faces(base64_str: str):
    try:
        header, encoded = base64_str.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
        img_np = np.array(img)[:, :, ::-1]  # Convert to BGR format for OpenCV

        print(f"📥 Received base64 image of length: {len(encoded)}")
        print(f"✅ Decoded image shape: {img_np.shape}")

        faces = app.get(img_np)
        if not faces:
            print("❌ No faces detected in the image.")
            return None
        print(f"✅ Detected {len(faces)} face(s) in the image.")
        
        face = faces[0]
        print(f"preparing model for ctx_id=0")
    
        app.prepare(ctx_id=0, det_size=(640, 640))  # Ensure model is prepared
        emb = app.get(img_np)[0].embedding
        print(f"✅ Extracted embedding of shape: {emb.shape}")
        if emb is None:
            print("❌ No embedding extracted.")
            return None
        # Ensure embedding is a numpy array
        if not isinstance(emb, np.ndarray):
            emb = np.array(emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        print(f"✅ Embedding shape after reshape: {emb.shape}")
        # Convert to float64 for consistency
        if emb.dtype != np.float64:
            print(f"Converting embedding from {emb.dtype} to float64")
            emb = emb.astype(np.float64)
        print(f"✅ Final embedding shape: {emb.shape}")
        # Return the embedding as a numpy array
        print("✅ Successfully decoded and aligned face.")  
        return emb.astype(np.float64)
    except Exception as e:
        print(f"❌ Error decoding or aligning faces: {e}")
        return None 


def extract_embedding(face_img: np.ndarray) -> np.ndarray:
    try:
        print("Face image shape:", face_img.shape)
        embedding = app.get_embedding(face_img)
        print("Extracted embedding:", embedding[:5])
        return embedding.astype(np.float64)
    except Exception as e:
        print(f"❌ Error extracting embedding: {e}")
        return np.zeros(512, dtype=np.float64)


