import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

def main():
    print("Downloading MNIST dataset from OpenML...")
    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Extract data and normalize pixel values to [0, 1]
    X_full = mnist.data / 255.0  
    labels_full = mnist.target.astype(int)

    print(f"Successfully loaded {X_full.shape[0]} images.")

    # ---------------------------------------------------------
    # RUN PCA (Directly to 2 Dimensions)
    # ---------------------------------------------------------
    print("Running PCA to reduce 784 dimensions down to 2...")
    t0 = time.time()
    
    # We ask PCA for exactly 2 components to match the t-SNE plot
    pca = PCA(n_components=2, random_state=42)
    Y_pca = pca.fit_transform(X_full)
    
    print(f"PCA completed in {time.time() - t0:.2f} seconds.")
    
    # Print out how much variance is actually captured by these 2 dimensions
    variance = pca.explained_variance_ratio_
    print(f"Variance explained by PC1: {variance[0]*100:.2f}%")
    print(f"Variance explained by PC2: {variance[1]*100:.2f}%")
    print(f"Total variance preserved: {np.sum(variance)*100:.2f}%")

    # ---------------------------------------------------------
    # PLOT THE RESULTS
    # ---------------------------------------------------------
    print("Plotting the 2D PCA embeddings...")
    plt.figure(figsize=(12, 10))
    
    # Scatter plot, color-coded by the true MNIST labels
    scatter = plt.scatter(Y_pca[:, 0], Y_pca[:, 1], c=labels_full, cmap='tab10', s=1, alpha=0.8)
    
    # Add a discrete colorbar
    cbar = plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    cbar.ax.set_yticklabels([str(i) for i in range(10)]) 
    
    plt.title("Standard PCA on Full MNIST (70,000 digits)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save a high-resolution copy for your research paper
    plt.savefig("mnist_full_pca.png", dpi=300, bbox_inches='tight')
    print("High-res plot saved as 'mnist_full_pca.png'.")
    
    plt.show()

if __name__ == "__main__":
    main()