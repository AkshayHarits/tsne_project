import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

# Paths
PCA_PATH = "data/pca_features.npy"
IDS_PATH = "data/clean_ids.npy"
CSV_PATH = "data/golden_sample.csv"

def run_tsne_clustering():
    if not os.path.exists(PCA_PATH):
        print(f"Error: {PCA_PATH} not found.")
        return

    # 1. Load the compressed data and IDs
    X_pca = np.load(PCA_PATH)
    valid_ids = np.load(IDS_PATH)
    
    print(f"Loaded PCA matrix: {X_pca.shape}")
    
    # 2. Load the catalog to get physical properties for coloring
    df = pd.read_csv(CSV_PATH)
    
    # Create dictionaries for fast lookup
    teff_dict = dict(zip(df['APOGEE_ID'].astype(str), df['TEFF']))
    feh_dict = dict(zip(df['APOGEE_ID'].astype(str), df['FE_H']))
    
    # Map properties exactly to the stars that survived preprocessing
    teff_values = np.array([teff_dict.get(str(star_id), np.nan) for star_id in valid_ids])
    feh_values = np.array([feh_dict.get(str(star_id), np.nan) for star_id in valid_ids])
    
    print("Running t-SNE... (This calculates the non-linear distances)")
    
    # --- UPDATED: Increased perplexity to 100 for large datasets and explicitly set max_iter ---
    tsne = TSNE(n_components=2, perplexity=100, max_iter=2000, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_pca)
    np.save("data/tsne_2d.npy", X_tsne)
    
    print("t-SNE Complete! Generating plots...")
    
    # --- PLOT 1: Colored by Temperature ---
    plt.figure(figsize=(10, 8))
    # 'plasma' is a great colormap for heat/temperature
    scatter1 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=teff_values, cmap='plasma', s=25, alpha=0.9)
    cbar1 = plt.colorbar(scatter1)
    cbar1.set_label('Effective Temperature (K) [TEFF]')
    
    plt.title("t-SNE Projection of APOGEE Spectra (Unsupervised Clustering)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/tsne_temperature.png")
    plt.show()

    # --- PLOT 2: Colored by Metallicity (Iron/Hydrogen ratio) ---
    plt.figure(figsize=(10, 8))
    # 'viridis' helps differentiate chemical abundances
    scatter2 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=feh_values, cmap='viridis', s=25, alpha=0.9)
    cbar2 = plt.colorbar(scatter2)
    cbar2.set_label('Metallicity [FE_H]')
    
    plt.title("t-SNE Projection of APOGEE Spectra (Colored by Metallicity)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/tsne_metallicity.png")
    plt.show()

    # --- NEW PLOT 3: Colored by Surface Gravity (Giant/Dwarf separation) ---
    logg_dict = dict(zip(df['APOGEE_ID'].astype(str), df['LOGG']))
    logg_values = np.array([logg_dict.get(str(star_id), np.nan) for star_id in valid_ids])

    plt.figure(figsize=(10, 8))
    # 'magma' is great for structural density
    scatter3 = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=logg_values, cmap='magma', s=25, alpha=0.9)
    cbar3 = plt.colorbar(scatter3)
    cbar3.set_label('Surface Gravity [LOGG]')
    
    plt.title("t-SNE Projection (Colored by Surface Gravity)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("data/tsne_logg.png")
    plt.show()

if __name__ == "__main__":
    run_tsne_clustering()