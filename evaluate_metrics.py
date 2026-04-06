import pandas as pd
import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
import os

def evaluate_csv(filename):
    """Reads a saved run and calculates quality metrics."""
    if not os.path.exists(filename):
        print(f"Skipping {filename}: File not found.")
        return

    print(f"\n--- Evaluating: {filename} ---")
    
    # 1. Load the data
    df = pd.read_csv(filename)
    
    # 2. Extract components based on our exact column structure
    # The first 784 columns are the original pixels
    X = df.iloc[:, :784].values
    
    # The next 2 columns (784 and 785) are the mapped coordinates
    Y = df.iloc[:, 784:786].values
    
    # The last column is the label
    labels = df.iloc[:, -1].values

    # 3. Calculate Trustworthiness
    # "Are points that are close in 2D also close in 784D?"
    print("Calculating Trustworthiness (n_neighbors=10)...")
    tw_score = trustworthiness(X, Y, n_neighbors=10)
    
    # 4. Calculate Silhouette Score
    # "How well separated are the different digit classes in the 2D map?"
    print("Calculating Silhouette Score...")
    sil_score = silhouette_score(Y, labels)

    # 5. Report
    print(f"-> Trustworthiness:  {tw_score:.4f} (Higher is better, closer to 1.0)")
    print(f"-> Silhouette Score: {sil_score:.4f} (Higher is better, range -1 to 1)")

def main():
    # List all the files you want to compare
    # You can add the nu=0.1 or nu=2.0 files here once your teammate runs them!
    files_to_check = [
        "data_mnist_2500_bhtsne.csv",
        "data_custom_tsne_nu_1.csv",
        "data_custom_tsne_nu_0.5.csv"
    ]

    for file in files_to_check:
        evaluate_csv(file)

if __name__ == "__main__":
    main()