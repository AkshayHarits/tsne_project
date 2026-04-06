import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():
    print("Downloading MNIST dataset from OpenML (this may take a minute)...")
    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Extract data and normalize pixel values to [0, 1]
    X_full = mnist.data / 255.0  
    labels_full = mnist.target.astype(int)

    print(f"Successfully loaded {X_full.shape[0]} images.")

    # ---------------------------------------------------------
    # PREPROCESSING: PCA Pre-filter
    # ---------------------------------------------------------
    print("Reducing dimensions to 50 using PCA...")
    t0 = time.time()
    
    # Keep top 50 principal components
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_full)
    
    print(f"PCA completed in {time.time() - t0:.2f} seconds.")

    # ---------------------------------------------------------
    # RUN THE ALGORITHM (scikit-learn Barnes-Hut)
    # ---------------------------------------------------------
    print("Running Barnes-Hut t-SNE on the full 70,000 dataset...")
    print("Grab a coffee, this will take a few minutes even with Barnes-Hut!")
    t_start = time.time()
    
    # Instantiate scikit-learn's TSNE
    tsne = TSNE(
        n_components=2, 
        perplexity=30.0, 
        method='barnes_hut', # O(N log N) scaling
        max_iter=1000,
        random_state=42,
        n_jobs=-1            # Use all available CPU cores
    )
    
    Y = tsne.fit_transform(X_pca)
    print(f"t-SNE completed in {time.time() - t_start:.2f} seconds.")

    # ---------------------------------------------------------
    # SINGLE FILE SAVING LOGIC (The "All-in-One" CSV)
    # ---------------------------------------------------------
    # Note: If this file exists, it will be completely overwritten.
    filename = "data_tsne_full_bh.csv"
    print(f"Preparing to save all 70,000 points to {filename}...")

    # Column name preparation
    pixel_cols = [f"pixel_{i}" for i in range(X_full.shape[1])]
    mapped_cols = ["tsne_1", "tsne_2"]
    
    # Create DataFrames for a structured merge
    df_original = pd.DataFrame(X_full, columns=pixel_cols)
    df_mapped = pd.DataFrame(Y, columns=mapped_cols)
    df_labels = pd.DataFrame(labels_full, columns=["label"])
    
    # Combine horizontally: Original Pixels | t-SNE Coords | Label
    final_df = pd.concat([df_original, df_mapped, df_labels], axis=1)

    # Save to CSV
    final_df.to_csv(filename, index=False)
    print(f"Successfully saved results to: {filename}")

    # ---------------------------------------------------------
    # PLOT THE RESULTS
    # ---------------------------------------------------------
    print("Plotting the low-dimensional embeddings...")
    plt.figure(figsize=(12, 10))
    
    # Scatter plot, color-coded by the true MNIST labels
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels_full, cmap='tab10', s=1, alpha=0.8)
    
    # Add a discrete colorbar
    cbar = plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    cbar.ax.set_yticklabels([str(i) for i in range(10)]) 
    
    plt.title("Scikit-Learn Barnes-Hut t-SNE on Full MNIST (Saved to CSV)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save a high-resolution copy for your research paper
    plt.savefig("mnist_full_tsne.png", dpi=300, bbox_inches='tight')
    print("High-res plot saved as 'mnist_full_tsne.png'.")
    
    plt.show()

if __name__ == "__main__":
    main()