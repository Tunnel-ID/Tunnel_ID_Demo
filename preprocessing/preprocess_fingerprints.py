import cv2
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA

# Path for saving processed data
processed_data_dir = r"C:\Users\ANKIT SRIVASTAVA\Desktop\login\fuzzy_schnoor_signature\data\raw\fingerprints"
os.makedirs(processed_data_dir, exist_ok = True)

def preprocess_fingerprints_as_float(image_paths):
    """
    Process multiple fingerprint images into high precison floating point 
    representations with PCA
    
    """
    all_flat_vectors = []
    
    # Process each fingerprint
    for image_path in image_paths:
        # Load the fingerprint image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Normalize pixel values to [0,1] with high precision
        normalized_image = image.astype(np.float64) / 255.0

        # Flatten the image into a 1D vector
        flat_vector = normalized_image.flatten()

        # Standardize with high precison
        mean = np.mean(flat_vector, dtype=np.float64)
        std = np.std(flat_vector, dtype = np.float64)
        standardized_vector = (flat_vector - mean) / (std + 1e-8)

        all_flat_vectors.append(standardized_vector)

    # Convert to  2D array for PCA
    all_flat_vectors = np.array(all_flat_vectors, dtype=np.float64)

    # Determine the valid number of components for PCA
    n_samples, n_features = all_flat_vectors.shape
    n_components = min(300, n_samples, n_features)

    # Apply PCA if more than one sample is available
    if n_samples > 1:
        pca_model = PCA(n_components=n_components)
        reduced_vectors = pca_model.fit_transform(all_flat_vectors)
    else:
        pca_model=None
        reduced_vectors=all_flat_vectors

    return reduced_vectors, pca_model

def main():
    # Paths to raw fingerprint images
    raw_data_dir = r"C:\Users\ANKIT SRIVASTAVA\Desktop\login\fuzzy_schnoor_signature\data\raw\fingerprints"
    image_files = [ os.path.join(raw_data_dir, file) for file in os.listdir(raw_data_dir) if file.endswith(".bmp")]

    if not image_files:
        raise ValueError("No fingerprint image files in the data/raw/fingerprints was found")
    
    # Preprocess fingerprints and save the results
    reduced_vectors, pca_model = preprocess_fingerprints_as_float(image_files)

    # Save processed vector with high precision
    for i, vector in enumerate(reduced_vectors, start=1):
        output_path = os.path.join(processed_data_dir, f"fingerprint_{i}_processed.npy")
        np.save(output_path, vector.astype(np.float64))
        print(f"Processed fingerprint saved to {output_path}")


    # Save PCA model if available
    if pca_model is not None:
        pca_model_path = os.path.join(processed_data_dir, "pca_model.pkl")
        with open(pca_model_path, "wb") as f:
            pickle.dump(pca_model, f)
        print(f"PCA model was saved to {pca_model_path}")
    else:
        print("PCA model was not applied (single image provided)")

if __name__ == "__main__":
    main()
