import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import os

# Paths
TSNE_PATH = "data/tsne_2d.npy"
IDS_PATH = "data/clean_ids.npy"
CSV_PATH = "data/golden_sample.csv"
OUTLIER_OUTPUT = "data/dbscan_outliers.csv"

def extract_anomalies():
    if not os.path.exists(TSNE_PATH):
        print(f"Error: {TSNE_PATH} not found. Did you add the save line to run_tsne.py?")
        return

    print("Loading t-SNE map coordinates...")
    X_tsne = np.load(TSNE_PATH)
    valid_ids = np.load(IDS_PATH)
    df = pd.read_csv(CSV_PATH)

    print("Running DBSCAN to find dense islands and isolate outliers...")
    # eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
    # min_samples is the number of samples in a neighborhood for a point to be considered as a core point.
    # You may need to tweak eps depending on how spread out your t-SNE map is!
    dbscan = DBSCAN(eps=3.5, min_samples=15)
    labels = dbscan.fit_predict(X_tsne)

    # Calculate stats
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Found {n_clusters} distinct clusters.")
    print(f"Isolated {n_noise} anomalous stars (Noise/Outliers)!")

    # --- PLOT: The DBSCAN Clusters ---
    plt.figure(figsize=(12, 10))
    
    # Plot the normal clusters in light gray
    core_mask = labels != -1
    plt.scatter(X_tsne[core_mask, 0], X_tsne[core_mask, 1], 
                c=labels[core_mask], cmap='tab20', s=15, alpha=0.6, label='Main Clusters')
    
    # Plot the Anomalies/Noise in bright red
    noise_mask = labels == -1
    plt.scatter(X_tsne[noise_mask, 0], X_tsne[noise_mask, 1], 
                color='red', s=30, edgecolor='black', zorder=5, label='Anomalies (Noise)')

    plt.title(f"DBSCAN Anomaly Extraction (Found {n_noise} Outliers)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/dbscan_clusters.png")
    plt.show()

    # --- SAVE THE OUTLIERS ---
    print("Cross-referencing anomalous IDs with main catalog...")
    outlier_ids = valid_ids[noise_mask]
    
    # Filter original dataframe for just the weird stars
    outlier_df = df[df['APOGEE_ID'].isin(outlier_ids)].copy()
    
    # Save to a new CSV for the next phase of your project
    outlier_df.to_csv(OUTLIER_OUTPUT, index=False)
    print(f"Successfully saved {len(outlier_df)} outliers to {OUTLIER_OUTPUT}!")
    print("These are the stars you should inspect manually or run a zoomed-in t-SNE on!")

if __name__ == "__main__":
    extract_anomalies()