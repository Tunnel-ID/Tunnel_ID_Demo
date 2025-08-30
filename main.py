from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
import os, base64
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import cv2
import binascii

from mtcnn import MTCNN
from uuid import UUID

# If your class file is face_landmark_processor.py, adjust the import accordingly
from preprocessing.face_landmark import FaceLandmarkProcessor
# from preprocessing.face_landmark_processor import FaceLandmarkProcessor
from supabase import create_client, Client




load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DIST_DIR = os.path.join(BASE_DIR, "key-flow-identity", "dist")
ASSETS_DIR = os.path.join(DIST_DIR, "assets")

app = Flask(__name__, static_folder=None)
CORS(app)

# Set up Supabase client
SUPABASE_URL = "https://lpubddkncvqyvkauudig.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImxwdWJkZGtuY3ZxeXZrYXV1ZGlnIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDQ2NjU3NywiZXhwIjoyMDcwMDQyNTc3fQ.Y24yOG7xKdZ4QjU5JIiHYtZEBtXTEOgdI_8Fg8I-1uo"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Singletons (avoid re-init per request)
flp = FaceLandmarkProcessor()
mtcnn = MTCNN()

def process_image(data_url: str) -> np.ndarray:
    b64_part = data_url.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_part)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Failed to decode image")
    return img

def cap_size(img_bgr: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img_bgr
    scale = max_side / float(s)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

@app.route("/process_face", methods=["POST", "OPTIONS"])
def process_face():
    if request.method == "OPTIONS":
        r = jsonify({"status": "ok", "message": "CORS preflight passed"})
        r.headers.add("Access-Control-Allow-Origin", "*")
        r.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        r.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return r

    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400

        base64_image = data["image"]
        email = data.get("email", "noemail")
        print("📥 base64 length:", len(base64_image))

        # 1) Decode + cap size (keep plenty of pixels)
        img_bgr = process_image(base64_image)
        print("decoded shape:", img_bgr.shape, "dtype:", img_bgr.dtype)
        img_bgr = cap_size(img_bgr, 1600)

        # 2) Detect face with MTCNN (same as applic.py)
        detections = mtcnn.detect_faces(img_bgr)
        print("MTCNN faces:", len(detections))

        if not detections:
            # Save debug and bail early
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            dbg = f"debug_no_face_mtcnn_{ts}.jpg"
            cv2.imwrite(dbg, img_bgr)
            print("⚠️ No faces by MTCNN. Saved:", dbg)
            return jsonify({"status": "error", "message": "No face detected"}), 422

        # Pick best detection
        det = max(detections, key=lambda d: d.get("confidence", 0))
        x, y, w, h = det["box"]
        x, y = max(x, 0), max(y, 0)
        face = img_bgr[y:y+h, x:x+w]
        # Some safety: minimum reasonable crop size
        if face.size == 0 or min(face.shape[:2]) < 60:
            return jsonify({"status": "error", "message": "Face too small"}), 422

        # 3) Pass cropped face to FaceLandmarkProcessor
        embedding = flp.process(face)  # 956-dim normalized vector expected

        if embedding is None:
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            dbg = f"debug_no_facemesh_{ts}.jpg"
            cv2.imwrite(dbg, face)
            print("⚠️ FaceMesh failed on crop. Saved:", dbg)
            return jsonify({"status": "error", "message": "No face detected"}), 422

        # 4) Optional: save input frame for traceability
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        fname = f"face_capture_{ts}_{email.replace('@','_')}.jpg"
        try:
            cv2.imwrite(fname, img_bgr)
        except Exception:
            pass

        print(f"✅ Embedding OK: len={len(embedding)}")
        return jsonify({
            "status": "success",
            "message": "Face processed successfully",
            "embedding": embedding.tolist()
        }), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in /process_face: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def base64_to_hex(b64_str: str) -> str:
    """Convert base64 string to hex string"""
    try:
        # Remove any base64 padding
        b64_str = b64_str.rstrip('=')
        # Decode base64 to bytes then convert to hex
        return binascii.hexlify(base64.b64decode(b64_str + '=' * (-len(b64_str) % 4))).decode('ascii')
    except Exception as e:
        print(f"Error converting base64 to hex: {e}")
        return ""

@app.route('/save_facekey', methods=['POST'])
def save_facekey():
    try:
        print("📥 Received /save_facekey request")
        data = request.get_json()
        
        email = data["email"]
        user_id = data["user_id"]
        print(f"📧 Email from request: {email}")
        print(f"🆔 User ID from request: {user_id}")

        enrollment_data = {
            "user_id": user_id,
            "email": email,
            "method": "face",
            "sketch_bytes": data["sketch"],  # base64; PostgREST will decode to bytea
            "g_a_hex": base64_to_hex(data["g_a"]),
            "beta_hex": base64_to_hex(data["beta"]),
            "tag_hex": base64_to_hex(data["tag"]),
            "version": "v1",
            "is_active": True,
            "device_info": {"source": "web"},
            "params_hash": None,
            "embedding_hash": None,
            "liveness_score": None
        }

        # Deactivate prior enrollments
        print("🛑 Deactivating previous active face enrollments...")
        supabase.table("enrollments") \
            .update({"is_active": False}) \
            .eq("user_id", user_id) \
            .eq("method", "face") \
            .eq("is_active", True) \
            .execute()

        # Insert new enrollment
        print("🆕 Inserting new enrollment...")
        response = supabase.table("enrollments").insert(enrollment_data).execute()
        print("✅ Insert successful")

        return jsonify({"status": "success", "data": response.data})

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"🔥 Error saving face key: {e}")
        return jsonify({
            "status": "error",
            "message": str(e),
            "code": getattr(e, "code", None)
        }), 500




# Healthcheck endpoint (for EC2 load balancer, monitoring, etc.)
@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "TunnelID Flask backend is healthy."})

# Serve static assets (JS, CSS, images, etc.)
@app.route("/assets/<path:filename>")
def serve_assets(filename):
    return send_from_directory(ASSETS_DIR, filename)

# Serve top-level static files (like favicon, manifest)
@app.route("/<filename>")
def serve_file(filename):
    file_path = os.path.join(DIST_DIR, filename)
    if os.path.exists(file_path):
        return send_from_directory(DIST_DIR, filename)
    return send_from_directory(DIST_DIR, "index.html")

# Serve React SPA (fallback routing for /auth, /face-capture, etc.)
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    full_path = os.path.join(DIST_DIR, path)
    print(f"Requested path: {path}")
    print(f"Resolved file path: {full_path}")
    if os.path.exists(full_path) and not os.path.isdir(full_path):
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, "index.html")

if __name__ == "__main__":
    print(f"🚀 Serving frontend from: {DIST_DIR}")
    app.run(host="0.0.0.0", port=5000, debug=True)
