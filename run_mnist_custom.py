import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Import the algorithm from your custom module
from tsne import tsne 

def main():
    print("Downloading MNIST dataset from OpenML (this may take a minute)...")
    # Fetch the MNIST dataset (as_frame=False ensures we get numpy arrays, not pandas DataFrames)
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Extract data and normalize pixel values to [0, 1] for better PCA stability
    X_full = mnist.data / 255.0  
    labels_full = mnist.target.astype(int)

    print("Randomly sampling 2,500 digits...")
    np.random.seed(42)  # Set seed so you get the exact same 2,500 digits every time you run it
    sample_indices = np.random.choice(X_full.shape[0], 2500, replace=False)
    
    X = X_full[sample_indices]
    labels = labels_full[sample_indices]

    # ---------------------------------------------------------
    # RUN THE ALGORITHM
    # ---------------------------------------------------------
    print("Running custom heavy-tailed t-SNE (nu = 1)...")
    print("Check your console for the Iteration/Error logs from the autograd engine!")
    
    # The tsne function processes X and outputs the 2D coordinates into Y
    Y = tsne(X, no_dims=2, initial_dims=50, perplexity=20.0, nu=1)

    # ---------------------------------------------------------
    # PLOT THE RESULTS
    # ---------------------------------------------------------
    print("Plotting the low-dimensional embeddings...")
    plt.figure(figsize=(10, 8))
    
    # Scatter plot, color-coded by the true MNIST labels
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=15, alpha=0.8)
    
    plt.colorbar(scatter, label="Digit Class")
    plt.title("Custom Heavy-Tailed t-SNE Projection (nu=1)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(f"custom_tsne_nu_1.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()