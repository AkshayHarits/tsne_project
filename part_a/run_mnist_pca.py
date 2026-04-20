import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd # Using pandas makes multi-column CSV saving much easier
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

def main():
    print("Downloading MNIST dataset from OpenML...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    X_full = mnist.data / 255.0  
    labels_full = mnist.target.astype(int)

    # ---------------------------------------------------------
    # RUN PCA
    # ---------------------------------------------------------
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=42)
    Y_pca = pca.fit_transform(X_full)
    
    # ---------------------------------------------------------
    # SINGLE FILE SAVING LOGIC (The "All-in-One" CSV)
    # ---------------------------------------------------------
    # CHANGE THIS NAME FOR DIFFERENT RUNS (e.g., 'data_tsne_nu05.csv')
    filename = "data_pca_full.csv" 
    
    print(f"Preparing to save data to {filename}...")

    # Create column names for the 784 original pixels
    pixel_cols = [f"pixel_{i}" for i in range(X_full.shape[1])]
    # Create column names for the 2 mapped dimensions
    mapped_cols = ["mapped_1", "mapped_2"]
    
    # Combine everything into a single DataFrame
    # Structure: [Original Pixels] | [Mapped Points] | [Label]
    df_original = pd.DataFrame(X_full, columns=pixel_cols)
    df_mapped = pd.DataFrame(Y_pca, columns=mapped_cols)
    df_labels = pd.DataFrame(labels_full, columns=["label"])
    
    # Concatenate horizontally
    final_df = pd.concat([df_original, df_mapped, df_labels], axis=1)

    # Save to CSV (index=False prevents an extra 'unnamed' column)
    # This will overwrite the file if it already exists
    final_df.to_csv(filename, index=False)
    print(f"Successfully saved all points to: {filename}")

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 10))
    plt.scatter(Y_pca[:, 0], Y_pca[:, 1], c=labels_full, cmap='tab10', s=1, alpha=0.8)
    plt.colorbar(ticks=range(10), label="Digit Class")
    plt.title("PCA Result Saved to CSV")
    plt.show()

if __name__ == "__main__":
    main()