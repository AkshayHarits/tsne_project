import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

# Import the algorithm from your custom module
from tsne import tsne 

def main():
    print("Downloading MNIST dataset from OpenML (this may take a minute)...")
    # Fetch the MNIST dataset
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Extract data and normalize pixel values
    X_full = mnist.data / 255.0  
    labels_full = mnist.target.astype(int)

    print("Randomly sampling 2,500 digits...")
    np.random.seed(42)  
    sample_indices = np.random.choice(X_full.shape[0], 2500, replace=False)
    
    X = X_full[sample_indices]
    labels = labels_full[sample_indices]

    # ---------------------------------------------------------
    # RUN THE ALGORITHM
    # ---------------------------------------------------------
    nu_value = 1 # Keeping nu=1 as per your code
    print(f"Running custom heavy-tailed t-SNE (nu = {nu_value})...")
    print("Check your console for the Iteration/Error logs from the autograd engine!")
    
    Y = tsne(X, no_dims=2, initial_dims=50, perplexity=20.0, nu=nu_value)

    # ---------------------------------------------------------
    # SINGLE FILE SAVING LOGIC (All-in-One CSV)
    # ---------------------------------------------------------
    # Update this filename manually when running different nu values
    filename = f"data_custom_tsne_nu_{nu_value}.csv"
    print(f"Saving all sampled and mapped points to {filename}...")

    pixel_cols = [f"pixel_{i}" for i in range(X.shape[1])]
    mapped_cols = ["custom_tsne_1", "custom_tsne_2"]
    
    # Create DataFrames
    df_original = pd.DataFrame(X, columns=pixel_cols)
    df_mapped = pd.DataFrame(Y, columns=mapped_cols)
    df_labels = pd.DataFrame(labels, columns=["label"])
    
    # Combine: [Original Pixels] | [Mapped Points] | [Label]
    final_df = pd.concat([df_original, df_mapped, df_labels], axis=1)

    # Save to CSV (Overwrites if exists)
    final_df.to_csv(filename, index=False)
    print(f"Successfully saved to: {filename}")

    # ---------------------------------------------------------
    # PLOT THE RESULTS
    # ---------------------------------------------------------
    print("Plotting the low-dimensional embeddings...")
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=15, alpha=0.8)
    
    plt.colorbar(scatter, label="Digit Class")
    plt.title(f"Custom Heavy-Tailed t-SNE Projection (nu={nu_value})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(f"custom_tsne_nu_{nu_value}.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()