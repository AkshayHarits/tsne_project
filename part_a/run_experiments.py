#
# run_experiments.py
#
# Master orchestrator script to automate the heavy-tailed t-SNE testbed.
#

import numpy as np
import pandas as pd
import time
from sklearn.datasets import fetch_openml                
from sklearn.decomposition import PCA

# Import custom modules
from tsne import tsne 
from evaluate_metrics import calculate_metrics
from plot_utils import save_tsne_plot

def main():
    print("Downloading MNIST dataset from OpenML (this may take a minute)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    
    # Extract data and normalize pixel values
    X_full = mnist.data / 255.0  
    labels_full = mnist.target.astype(int)

    print("Randomly sampling 2,500 digits...")
    np.random.seed(42)  
    sample_indices = np.random.choice(X_full.shape[0], 2500, replace=False)
    
    X_sample = X_full[sample_indices]
    labels_sample = labels_full[sample_indices]

    # ---------------------------------------------------------
    # GLOBAL PREPROCESSING (Done ONCE to save time)
    # ---------------------------------------------------------
    print("Pre-computing PCA (reducing 784D -> 50D) for the entire loop...")
    pca = PCA(n_components=50, random_state=42)
    # X_pca is what we feed to t-SNE. X_sample is kept for Trustworthiness calc.
    X_pca = pca.fit_transform(X_sample)

    # ---------------------------------------------------------
    # TESTBED CONFIGURATION
    # Add or remove parameters here to automate the pipeline!
    # ---------------------------------------------------------
    distributions = ['t-distribution', 'power-law', 'gaussian']
    nu_values = [0.5, 1.0, 2.0]
    perplexities = [10.0, 20.0, 30.0, 50.0]

    total_runs = len(distributions) * len(nu_values) * len(perplexities)
    current_run = 0

    # ---------------------------------------------------------
    # THE ORCHESTRATOR LOOP
    # ---------------------------------------------------------
    for dist in distributions:
        for nu in nu_values:
            
            # The gaussian distribution doesn't actually use nu, 
            # so we can skip redundant runs to save time.
            if dist == 'gaussian' and nu != 1.0:
                continue 

            for perp in perplexities:
                current_run += 1
                print(f"\n========================================================")
                print(f"RUN {current_run}/{total_runs} | Dist: {dist} | nu={nu} | perp={perp}")
                print(f"========================================================")
                
                t_start = time.time()

                # 1. Run the custom Engine (Pass the PCA-reduced data)
                # Note: tsne.py will detect it's already 50D and skip internal PCA.
                # 1. Run the custom Engine
                if dist == 'gaussian':
                    # Ask for snapshots right before exaggeration ends (99) and shortly after (150)
                    snapshots = tsne(X_pca, no_dims=2, initial_dims=50, perplexity=perp, nu=nu, dist_type=dist, snapshot_iters=[99, 150])
                    
                    # Plot the intermediate snapshots!
                    for iter_step, Y_snap in snapshots.items():
                        if iter_step == 'final':
                            continue # We will plot the final one normally below
                        
                        snap_filename = f"results_{dist}_nu{nu}_p{perp}_ITER_{iter_step}.png"
                        # We don't calculate TW/SIL for intermediate steps to save time, just plot them
                        save_tsne_plot(Y_snap, labels_sample, dist, nu, perp, 0.0, 0.0, snap_filename)
                    
                    # Set Y to the final coordinates so the rest of the script works normally
                    Y = snapshots['final']
                else:
                    # Normal run for everything else
                    Y = tsne(X_pca, no_dims=2, initial_dims=50, perplexity=perp, nu=nu, dist_type=dist)
               
                # 2. Evaluate Metrics
                # Trustworthiness must be calculated using the original 784D space!
                tw_score, sil_score = calculate_metrics(X_sample, Y, labels_sample)
                
                print(f"-> FINAL SCORES | TW: {tw_score:.3f} | SIL: {sil_score:.3f}")

                # 3. Dynamic File Naming
                base_name = f"results_{dist}_nu{nu}_p{perp}"
                csv_filename = f"{base_name}.csv"
                plot_filename = f"{base_name}.png"

                # 4. Save the Plot
                save_tsne_plot(Y, labels_sample, dist, nu, perp, tw_score, sil_score, plot_filename)

                # 5. Save the CSV (All-in-One Format)
                print(f"Saving data to {csv_filename}...")
                pixel_cols = [f"pixel_{i}" for i in range(X_sample.shape[1])]
                mapped_cols = [f"custom_tsne_{dist}_1", f"custom_tsne_{dist}_2"]
                
                df_original = pd.DataFrame(X_sample, columns=pixel_cols)
                df_mapped = pd.DataFrame(Y, columns=mapped_cols)
                df_labels = pd.DataFrame(labels_sample, columns=["label"])
                
                final_df = pd.concat([df_original, df_mapped, df_labels], axis=1)
                final_df.to_csv(csv_filename, index=False)
                
                print(f"Run {current_run} complete in {time.time() - t_start:.2f} seconds.")

    print("\n ALL EXPERIMENTS COMPLETED SUCCESSFULLY.")

if __name__ == "__main__":
    main()