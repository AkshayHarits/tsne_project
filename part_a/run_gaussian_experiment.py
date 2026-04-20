import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

# Import your custom modules
from tsne import tsne 
import evaluate_metrics
import plot_utils

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
    # ALGORITHM TRACKING VARIABLES
    # (These must match what you manually typed inside tsne.py)
    # ---------------------------------------------------------
    nu_value = 1.0  
    perplexity_value = 30.0
    early_exag = 12.0
    distribution = "gaussian"

    print(f"\n--- Running Custom Autograd t-SNE ---")
    print(f"Tracking: Dist: {distribution} | Perp: {perplexity_value} | Exag: {early_exag}")
    print("⚠️ REMINDER: Ensure you manually set exaggeration to 12 in your tsne.py file! ⚠️")
    
    # Run the custom engine (arguments match your existing tsne.py signature)
    Y = tsne(
        X, 
        no_dims=2, 
        initial_dims=50, 
        perplexity=perplexity_value, 
        nu=nu_value
    )

    # ---------------------------------------------------------
    # EVALUATE METRICS
    # ---------------------------------------------------------
    print("\n--- Evaluating Quality Metrics ---")
    tw_score, sil_score = evaluate_metrics.calculate_metrics(X, Y, labels)
    print(f"Final TW: {tw_score:.3f} | Final SIL: {sil_score:.3f}")

    # ---------------------------------------------------------
    # SAVE CSV
    # ---------------------------------------------------------
    filename_csv = f"data_{distribution}_perp_{perplexity_value}_exag_{early_exag}.csv"
    print(f"\nSaving mapped points to {filename_csv}...")

    pixel_cols = [f"pixel_{i}" for i in range(X.shape[1])]
    mapped_cols = ["custom_tsne_1", "custom_tsne_2"]
    
    df_original = pd.DataFrame(X, columns=pixel_cols)
    df_mapped = pd.DataFrame(Y, columns=mapped_cols)
    df_labels = pd.DataFrame(labels, columns=["label"])
    
    final_df = pd.concat([df_original, df_mapped, df_labels], axis=1)
    final_df.to_csv(filename_csv, index=False)

    # ---------------------------------------------------------
    # PLOT USING PLOT_UTILS
    # ---------------------------------------------------------
    filename_jpg = f"results_{distribution}_perp_{perplexity_value}_exag_{early_exag}_FINAL.jpg"
    
    plot_utils.save_tsne_plot(
        Y, 
        labels, 
        dist_type=distribution, 
        nu=nu_value, 
        perp=perplexity_value, 
        tw_score=tw_score, 
        sil_score=sil_score, 
        filename=filename_jpg
    )
    
    print(f"Run complete! Check {filename_jpg} to see if the hairball separated.")

if __name__ == "__main__":
    main()