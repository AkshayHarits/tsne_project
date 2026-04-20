import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

# Paths
MATRIX_PATH = "data/clean_fluxes.npy"
OUTPUT_PCA = "data/pca_features.npy"

def compress_spectra():
    if not os.path.exists(MATRIX_PATH):
        print(f"Error: {MATRIX_PATH} not found.")
        return

    print("Loading clean spectra matrix...")
    flux_matrix = np.load(MATRIX_PATH)
    
    print(f"Original shape: {flux_matrix.shape} (Stars, Pixels)")
    
    # 1. Standardize the data (Mean centering is required for PCA)
    # We subtract the mean spectrum from every star
    mean_spectrum = np.mean(flux_matrix, axis=0)
    centered_matrix = flux_matrix - mean_spectrum

    # 2. Run PCA
    # We will compress 8,575 pixels down to 50 principal components
    n_components = 50
    print(f"Running PCA to compress to {n_components} components...")
    
    pca = PCA(n_components=n_components, random_state=42)
    pca_features = pca.fit_transform(centered_matrix)

    # 3. Calculate how much "information" we kept
    variance_retained = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Total Variance Retained: {variance_retained:.2f}%")

    # 4. Save the compressed ML-ready matrix
    np.save(OUTPUT_PCA, pca_features)
    print(f"Saved compressed matrix: {pca_features.shape} to {OUTPUT_PCA}")

    # 5. Plot the Explained Variance (Great for your report!)
    plt.figure(figsize=(8, 5))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-', color='b')
    plt.title("PCA: Cumulative Explained Variance")
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Retained")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Information')
    plt.legend()
    plt.tight_layout()

    # 5.5 Plot the Top 3 Eigenvectors (Physical Sanity Check)
    plt.figure(figsize=(14, 6))
    plt.title("Top 3 Principal Components (Eigen-spectra)")
    # We plot a small 300-pixel window to actually see the lines
    plt.plot(pca.components_[0][1000:1300], label='PC1 (Primary Variance)', alpha=0.8)
    plt.plot(pca.components_[1][1000:1300], label='PC2', alpha=0.8)
    plt.plot(pca.components_[2][1000:1300], label='PC3', alpha=0.8)
    plt.xlabel("Pixel Index")
    plt.ylabel("Component Weight")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Show both plots!
    plt.show()

if __name__ == "__main__":
    compress_spectra()