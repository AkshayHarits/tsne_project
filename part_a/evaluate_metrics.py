#
# evaluate_metrics.py
#
# Modular script to calculate clustering quality metrics.
#

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score
import os
import pandas as pd

def calculate_metrics(X_high_dim, Y_low_dim, labels):
    """
    Calculates Trustworthiness and Silhouette Score from memory.
    
    Parameters:
    - X_high_dim: The original 784D pixel data.
    - Y_low_dim: The 2D mapped coordinates from the t-SNE engine.
    - labels: The ground truth digit labels.
    
    Returns:
    - tw_score (float), sil_score (float)
    """
    print("Calculating Trustworthiness (n_neighbors=10)...")
    tw_score = trustworthiness(X_high_dim, Y_low_dim, n_neighbors=10)
    
    print("Calculating Silhouette Score...")
    # Silhouette score requires at least 2 clusters, which MNIST has (10).
    sil_score = silhouette_score(Y_low_dim, labels)
    
    return tw_score, sil_score

def evaluate_csv(filename):
    """
    Legacy function: Reads a saved run and calculates quality metrics from disk.
    Kept for backward compatibility and manual checks.
    """
    if not os.path.exists(filename):
        print(f"Skipping {filename}: File not found.")
        return

    print(f"\n--- Evaluating: {filename} ---")
    df = pd.read_csv(filename)
    
    X = df.iloc[:, :784].values
    Y = df.iloc[:, 784:786].values
    labels = df.iloc[:, -1].values

    tw_score, sil_score = calculate_metrics(X, Y, labels)

    print(f"-> Trustworthiness:  {tw_score:.4f} (Higher is better, closer to 1.0)")
    print(f"-> Silhouette Score: {sil_score:.4f} (Higher is better, range -1 to 1)")