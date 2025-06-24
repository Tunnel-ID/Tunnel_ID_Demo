from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import csv
import numpy as np
import base64
import cv2
from mtcnn import MTCNN
from datetime import datetime
from sklearn.decomposition import PCA
import binascii
import logging
from dotenv import load_dotenv
from preprocessing.face import decode_and_align_faces
import base64
import json

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# ✅ Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
##// Uncomment for production
#app = Flask(__name__)
app = Flask(__name__, static_folder="/home/canna/Documents/TunnelID/Tunnel-ID/Frontend/dist", static_url_path="")
#app = Flask(__name__)
# Enable CORS
# CORS(app, origins="http://localhost:5173")
CORS(app, resources={r"/*": {"origins": [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5001",
    "https://tunnelid.me",
    "https://www.tunnelid.me"
]}})
#CORS(app, resources={r"/*": {"origins": ["http://localhost:5173", "http://localhost:3000","https://tunnelid.me", "https://www.tunnelid.me"]}})
# Secret key for session management
# Allow both localhost and 127.0.0.1
# CORS(app,  supports_credentials=True)
app.secret_key = "supersecretkey"

# Shared configurations
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
FINGERPRINT_DIR = os.path.join(PROCESSED_DIR, "fingerprints")
FACIAL_DATA_DIR = os.path.join(PROCESSED_DIR, "facial_data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_FINGERPRINTS_DIR = os.path.join(RAW_DIR, "fingerprints")
DATABASE_CSV = os.path.join(BASE_DIR, "database.csv")

# Ensure directories exist
os.makedirs(FINGERPRINT_DIR, exist_ok=True)
os.makedirs(FACIAL_DATA_DIR, exist_ok=True)
os.makedirs(RAW_FINGERPRINTS_DIR, exist_ok=True)

# Define OUTPUT_DIR for storing captured faces
OUTPUT_DIR = FACIAL_DATA_DIR

# Ensure the unified CSV file exists with headers
if not os.path.exists(DATABASE_CSV):
    with open(DATABASE_CSV, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Sketch_Numpy", "Sketch_Base64", "Sketch_Hex", "Wallet_Address", "Type_of_Credential", "Face_Filenames"])

# Initialize MTCNN for face detection
face_detector = MTCNN()


# LinearSketch class
class LinearSketch:
    def __init__(self, basis_vectors, modulus, threshold=5.0):
        self.basis_vectors = basis_vectors
        self.modulus = modulus
        self.threshold = threshold

    def sketch(self, data):
        return np.dot(data, self.basis_vectors) % self.modulus
    
# ✅ Initialize LinearSketch
basis_vectors = [[1, 0], [0.5, np.sqrt(3) / 2]]
modulus = 7
linear_sketch = LinearSketch(basis_vectors, modulus, 5.0)

# Function to process Base64 image data
def process_image(image_data):
    try:
        print("Processing image data...")
        image_data = base64.b64decode(image_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        print("Image processed successfully")
        return image
    except Exception as e:
        print(f"Error processing image: {e}")
        raise

# Function to capture faces and store them in output_dir
def capture_faces_from_image(image):
    try:
        print("Detecting faces...")
        faces = face_detector.detect_faces(image)
        print(f"Number of faces detected: {len(faces)}")
        if len(faces) == 0:
            raise ValueError("No faces detected in the image")

        captured_faces = []
        filenames = []

        for i, face in enumerate(faces):
            x, y, w, h = face['box']
            face_img = image[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (128, 128)).astype(np.float64) / 255.0
            
            # Generate filename for face data
            filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}.npy"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Save the face as a .npy file
            np.save(filepath, face_resized.flatten())

            captured_faces.append(face_resized.flatten())
            filenames.append(filename)

        return np.array(captured_faces), filenames
    except Exception as e:
        print(f"Error capturing faces: {e}")
        raise

# Function to store data in the unified CSV file
def store_in_database(sketch_numpy, sketch_base64, sketch_hex, wallet_address, type_of_credential, face_filenames=None):
    try:
        print("Storing data in database.csv...")
        with open(DATABASE_CSV, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([sketch_numpy.tolist(), sketch_base64, sketch_hex, wallet_address, type_of_credential, ", ".join(face_filenames) if face_filenames else ""])
        print("Data stored successfully")
    except Exception as e:
        print(f"Error storing data: {e}")
        raise

# Function to verify sketch against the unified CSV file
def verify_sketch_id(sketch_base64):
    if not os.path.exists(DATABASE_CSV):
        return False  # File doesn't exist, so no match found

    with open(DATABASE_CSV, mode="r", newline="") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            if sketch_base64 in row:  # Check if the Sketch ID exists
                return True
    return False

# Serve frontend

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "success", "message": "Hello! Flask API is running!"}), 200

@app.errorhandler(404)
def not_found(error):
    return send_from_directory(app.static_folder, "index.html")  

@app.route('/')
def index():
    return send_from_directory(app.static_folder, "index.html")
@app.route('/<path:path>')
def catch_all(path):
    full_path = os.path.join(app.static_folder, path)
    if os.path.exists(full_path):
        return send_from_directory(app.static_folder, path) 
    
    return send_from_directory(app.static_folder, "index.html")  # Fallback to index.html for SPA routing
# Fingerprint registration endpoint
@app.route("/register_fingerprint", methods=["POST"])
def register_fingerprint():
    try:
        fingerprint_file = request.files.get("fingerprint_file")
        wallet_address = request.form.get("wallet_address")

        if not fingerprint_file or not wallet_address:
            return jsonify({"error": "Fingerprint file and wallet address are required!"}), 400

        fingerprint = np.load(fingerprint_file)
        sketch = linear_sketch.sketch(fingerprint)

        # Convert sketch to Base64
        sketch_bytes = np.array(sketch).tobytes()
        sketch_b64 = base64.b64encode(sketch_bytes).decode("utf-8")

        # Convert sketch to 64 bytes and then to hexadecimal
        sketch_64bytes = sketch_bytes.ljust(64, b'\0')  # Ensure it's 64 bytes
        sketch_hex = binascii.hexlify(sketch_64bytes).decode("utf-8")

        # Check if user already exists
        with open(DATABASE_CSV, mode="r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[1] == sketch_b64:
                    return jsonify({"error": "User already exists!"}), 400

        # Find the next serial number for the fingerprint file
        existing_files = [f for f in os.listdir(FINGERPRINT_DIR) if f.endswith("fingerprint.npy")]
        numbers = [int(f.split("fingerprint.npy")[0]) for f in existing_files if f[:-14].isdigit()]
        next_number = max(numbers) + 1 if numbers else 1

        # Define fingerprint file path
        fingerprint_path = os.path.join(FINGERPRINT_DIR, f"{next_number}fingerprint.npy")
        np.save(fingerprint_path, fingerprint)

        # Save to CSV
        store_in_database(sketch, sketch_b64, sketch_hex, wallet_address, "fingerprint")

        return jsonify({
            "message": "Fingerprint registration successful!",
            "user_id": sketch_b64,
            "fingerprint_path": fingerprint_path,
            "wallet_address": wallet_address,
            "sketch_hex": sketch_hex
        }), 200
    except Exception as e:
        print(f"Error in /register_fingerprint: {e}")
        return jsonify({"error": str(e)}), 500

# Facial registration endpoint
@app.route('/register_facial', methods=['POST'])
def register_facial():
    try:
        # ✅ Get two facial images from frontend
        data = request.json
        image_data_1 = data.get("image1")
        image_data_2 = data.get("image2")
        print("image1 present:", bool(image_data_1))
        print("image2 present", bool(image_data_2))

        if not image_data_1 or not image_data_2:
            return jsonify({"status": "error", "message": "Both facial images are required."}), 400

        print("Received image1 and image2")

        # ✅ Decode both images
        try:
            image1 = process_image(image_data_1)
            image2 = process_image(image_data_2)
        except Exception as e:
            print(f"Error processing images: {e}")
            return jsonify({"status": "error", "message": "Failed to process one or both images."}), 400

        # ✅ Extract one face from each image
        faces = []
        face_filenames = []

        for img in [image1, image2]:
            detected_faces, filenames = capture_faces_from_image(img)
            if len(detected_faces) == 0:
                return jsonify({"status": "error", "message": "No face detected in one of the images."}), 400
            faces.append(detected_faces[0])  # Take first detected face
            face_filenames.extend(filenames)

        faces = np.array(faces)

        # ✅ Train PCA with 2 samples
        n_components = min(2, faces.shape[1])
        print(f"Using n_components={n_components} for PCA")
        try:
            pca = PCA(n_components=n_components)
            pca.fit(faces)
            print("PCA trained successfully")
        except Exception as e:
            print(f"Error training PCA: {e}")
            return jsonify({"status": "error", "message": "Failed to train PCA."}), 500

        # ✅ Transform first face only (you could average both if desired)
        try:
            transformed_data = pca.transform(faces[0].reshape(1, -1))
            print(f"Transformed data shape: {transformed_data.shape}")
        except Exception as e:
            print(f"Error transforming data with PCA: {e}")
            return jsonify({"status": "error", "message": "Failed to transform data."}), 500

        # ✅ Pad to 2D if needed
        if transformed_data.shape[1] < 2:
            padding = np.zeros((transformed_data.shape[0], 2 - transformed_data.shape[1]))
            transformed_data = np.hstack((transformed_data, padding))
            print(f"Padded transformed data shape: {transformed_data.shape}")

        # ✅ Generate sketch
        try:
            new_sketch = linear_sketch.sketch(transformed_data.flatten())
            print(f"Generated sketch: {new_sketch}")
        except Exception as e:
            print(f"Error generating sketch: {e}")
            return jsonify({"status": "error", "message": "Failed to generate sketch."}), 500

        # ✅ Encode sketch
        sketch_base64 = base64.b64encode(new_sketch.astype(np.float64)).decode('utf-8')
        sketch_hex = new_sketch.astype(np.float64).tobytes().hex()

        print(f"Sketch (Base64): {sketch_base64}")
        print(f"Sketch (Hex): {sketch_hex}")

        # ✅ Check if user already exists
        try:
            with open(DATABASE_CSV, mode="r", newline="") as file:
                reader = csv.reader(file)
                next(reader)
                for row in reader:
                    if row and row[1] == sketch_base64:
                        return jsonify({"status": "error", "message": "User already exists!"}), 400
        except Exception as e:
            print(f"Error checking for existing user: {e}")
            return jsonify({"status": "error", "message": "Failed to check for existing user."}), 500

        # ✅ Store in database
        try:
            store_in_database(new_sketch, sketch_base64, sketch_hex, None, "facial", face_filenames)
        except Exception as e:
            print(f"Error storing in database: {e}")
            return jsonify({"status": "error", "message": "Failed to store sketch."}), 500

        return jsonify({
            "status": "success",
            "message": "Facial registration successful.",
            "sketch_base64": sketch_base64,
            "sketch_hex": sketch_hex,
            "face_filenames": face_filenames
        }), 200

    except Exception as e:
        print(f"Unexpected error in /register_facial: {e}")
        return jsonify({"status": "error", "message": "Unexpected server error."}), 500

# Fingerprint verification endpoint
@app.route('/verify_sketch', methods=['POST'])
def verify_sketch():
    try:
        data = request.json
        sketch_b64 = data.get('sketch_b64')

        if not sketch_b64:
            return jsonify({"status": "error", "message": "No sketch_b64 provided."}), 400

        # Check if the sketch exists in database.csv
        if not os.path.exists(DATABASE_CSV):
            return jsonify({"status": "error", "message": "Database not found."}), 404

        with open(DATABASE_CSV, mode="r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header row
            for row in reader:
                if row and row[1] == sketch_b64:  # Check if sketch_b64 exists in column 1
                    return jsonify({"status": "success", "message": "✅ Verified! Sketch ID exists."}), 200

        return jsonify({"status": "error", "message": "❌ Sketch ID not found."}), 404

    except Exception as e:
        print(f"Error in /verify_sketch: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    
# Facial verification endpoint
@app.route('/verify', methods=['POST'])
def verify_facial():
    try:
        data = request.json
        sketch_base64 = data.get('sketch_base64')

        if not sketch_base64:
            return jsonify({"status": "error", "message": "No Sketch ID provided."}), 400

        # Check if the Sketch ID exists in the unified CSV file
        if verify_sketch_id(sketch_base64):
            return jsonify({"status": "success", "message": "✅ Verified! User already exists."}), 200
        else:
            return jsonify({"status": "error", "message": "❌ Sketch ID not found."}), 404

    except Exception as e:
        print(f"Error in /verify_facial: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    
@app.route("/admin/clear_facial_data", methods=["POST"])
def clear_facial_data():
    try:
        # Remove all .npy files from facial data and raw folders
        for directory in [FACIAL_DATA_DIR, RAW_DIR]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if filename.endswith(".npy") and os.path.isfile(file_path):
                    os.remove(file_path)

        # Clear only "facial" entries from the database.csv
        rows_to_keep = []
        with open(DATABASE_CSV, mode="r", newline="") as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                if row and row[4] != "facial":
                    rows_to_keep.append(row)

        with open(DATABASE_CSV, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            writer.writerows(rows_to_keep)

        return jsonify({"status": "success", "message": "Facial data cleared."}), 200

    except Exception as e:
        print(f"Error in /admin/clear_facial_data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

## Face detection endpoint
@app.route("/process_face", methods=["POST", "OPTIONS"])
def process_face():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok", "message": "CORS preflight passed"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        return response
    try:
        data = request.get_json(force=True)
        print("Received data:", data)
        # Check if the request contains image data  

        if not data or "image" not in data:
            return jsonify({"status": "error", "message": "No image data provided"}), 400
        
        base64_image = data["image"]
        print("📥 Received base64 image of length:", len(base64_image))

        embedding = decode_and_align_faces(base64_image)
        if embedding is None:
            print("❌ Returning error: No face detected.")
            return jsonify({"status": "error", "message": "No face detected"}), 400
        
        
        
        return jsonify({
            "status": "success",
            "message": "Face processed successfully",
            "embedding": embedding.tolist()  # Convert numpy array to list for JSON serialization
        }), 200
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in /process_face: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def decode_sketch_base64(sketch_base64):
    try:
        raw_bytes = base64.b64decode(sketch_base64)
        sketch_array = np.frombuffer(raw_bytes, dtype=np.float64)
        if not (128 <= len(sketch_array) <= 1024):
            raise ValueError(f"Sketch length invalid: {len(sketch_array)}")
        return sketch_array.tolist()
    except Exception as e:
        raise ValueError(f"Invalid sketch format: {e}")

    
@app.route("/save_facekey", methods=["POST"])
def save_facial_credentail():
    try:
        data = request.get_json(force=True)
        if not data:   
            return jsonify({"status": "error", "message": "No data provided"}), 400
        
        print("📬 Received data for saving facial credential:", data)

        wallet_address = data.get("wallet_address")
        sketch_b64 = data.get("sketch_base64")
        g_a_b64 = data.get("g_a")
        beta_b64 = data.get("beta")

        if not wallet_address or not sketch_b64 or not g_a_b64 or not beta_b64:
            return jsonify({"status": "error", "message": "Wallet address, sketch_base64, g_a, and beta are required"}), 400
        print("📬 Received wallet address for saving facial credential:", wallet_address)

        try:
            sketch_array = decode_sketch_base64(sketch_b64)
        except ValueError as ve:
            print(f"❌ Error decoding sketch_base64: {ve}")
            return jsonify({"status": "error", "message": str(ve)}), 400
        print("✅ Successfully decoded sketch_base64")

        try:
            g_a_bytes = base64.b64decode(g_a_b64)
            beta_bytes = base64.b64decode(beta_b64)

        except Exception as e:
            print(f"❌ Error decoding g_a or beta: {e}")
            return jsonify({"status": "error", "message": "Invalid g_a or beta format"}), 400
        
        if len(g_a_bytes) != 65:
            return jsonify({"status": "error", "message": "g_a must be 65 bytes"}), 400
        if len(beta_bytes) != 32:
            return jsonify({"status": "error", "message": "beta must be 32 bytes"}), 400
        print("✅ Successfully decoded g_a and beta")

        file_path = "facial_credential.csv"
        fieldnames = ["Wallet_Address", "Sketch_Base64", "g_a", "beta"]

        # ✅ Check if wallet already exists
        if os.path.exists(file_path):
            with open(file_path, mode="r", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row["Wallet_Address"] == wallet_address:
                        return jsonify({"status": "error", "message": "Wallet address already exists"}), 400

        # ✅ Append new credential
        file_exists = os.path.exists(file_path)
        with open(file_path, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            if not file_exists or os.stat(file_path).st_size == 0:
                writer.writeheader()
            writer.writerow({
                "Wallet_Address": wallet_address,
                "Sketch_Base64": sketch_b64,
                "g_a": g_a_b64,
                "beta": beta_b64
            })

        print(f"✅ Credential saved for {wallet_address}")
        return jsonify({"status": "success", "message": "Credential saved successfully"}), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


        
    

@app.route("/recover_facekey", methods=["POST"])
def get_facekey_by_wallet():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "message": "No data provided"}), 400
        tag = data.get("tag")
        if not tag:
            return jsonify({"status": "error", "message": "Tag is required"}), 400
        print("📬 Received tag for recovering facekey:", tag)

        with open("facial_credential.csv", mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["Wallet_Address"] == tag:
                    print("✅ Facial credential found for wallet address:", tag)
                    return jsonify({
                        "status": "success",
                        "message": "Facial credential found",
                        "sketch_base64": row["Sketch_Base64"],
                        "g_a": row["g_a"],
                        "beta": row["beta"]
                    }), 200
                
        print("❌ Facial credential not found for wallet address:", tag)
        return jsonify({"status": "error", "message": "Facial credential not found"}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in /recover_facekey: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
            

# Run the Flask app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)
# if __name__ == "__main__":
#     app.run(debug=True, port=5001)
