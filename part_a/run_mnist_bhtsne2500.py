import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def main():
    print("Downloading MNIST dataset from OpenML...")
    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Extract data and normalize pixel values to [0, 1]
    X_full = mnist.data / 255.0  
    labels_full = mnist.target.astype(int)

    # ---------------------------------------------------------
    # NEW STEP: Sampling 2,500 points for fair comparison
    # ---------------------------------------------------------
    print("Randomly sampling 2,500 digits for comparison...")
    np.random.seed(42) # Using same seed as custom script for consistency
    sample_indices = np.random.choice(X_full.shape[0], 2500, replace=False)
    
    X = X_full[sample_indices]
    labels = labels_full[sample_indices]

    # ---------------------------------------------------------
    # PREPROCESSING: PCA Pre-filter
    # ---------------------------------------------------------
    print("Reducing dimensions to 50 using PCA...")
    t0 = time.time()
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X) # Run PCA on the 2500 sample
    print(f"PCA completed in {time.time() - t0:.2f} seconds.")

    # ---------------------------------------------------------
    # RUN THE ALGORITHM (scikit-learn Barnes-Hut)
    # ---------------------------------------------------------
    print("Running Barnes-Hut t-SNE on 2,500 points...")
    t1 = time.time()
    
    tsne_model = TSNE(
        n_components=2, 
        perplexity=20.0,      # Matching the perplexity of your custom run
        method='barnes_hut', 
        max_iter=1000,
        random_state=42,
        n_jobs=-1 
    )
    
    Y = tsne_model.fit_transform(X_pca)
    print(f"t-SNE completed in {time.time() - t1:.2f} seconds.")

    # ---------------------------------------------------------
    # SINGLE FILE SAVING LOGIC (All-in-One CSV)
    # ---------------------------------------------------------
    filename = "data_mnist_2500_bhtsne.csv"
    print(f"Saving sampled and mapped points to {filename}...")

    # Define column names
    pixel_cols = [f"pixel_{i}" for i in range(X.shape[1])]
    mapped_cols = ["tsne_1", "tsne_2"]
    
    # Create DataFrames for the original pixels, the 2D output, and labels
    df_original = pd.DataFrame(X, columns=pixel_cols)
    df_mapped = pd.DataFrame(Y, columns=mapped_cols)
    df_labels = pd.DataFrame(labels, columns=["label"])
    
    # Concatenate horizontally: [784 Pixels] | [2 t-SNE Coords] | [1 Label]
    final_df = pd.concat([df_original, df_mapped, df_labels], axis=1)

    # Save to CSV (Overwrites if exists)
    final_df.to_csv(filename, index=False)
    print(f"Successfully saved to: {filename}")

    # ---------------------------------------------------------
    # PLOT THE RESULTS
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=15, alpha=0.8)
    
    cbar = plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    cbar.ax.set_yticklabels([str(i) for i in range(10)]) 
    
    plt.title("Sklearn Barnes-Hut t-SNE (2,500 Sample, nu=1.0)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig("mnist_2500_bhtsne.png", dpi=300, bbox_inches='tight')
    print("High-res plot saved as 'mnist_2500_bhtsne.png'.")
    plt.show()

if __name__ == "__main__":
    main()