#
# plot_utils.py
#
# Modular plotting script for dimensionality reduction outputs.
#

import matplotlib.pyplot as plt

def save_tsne_plot(Y, labels, dist_type, nu, perp, tw_score, sil_score, filename):
    """
    Generates and saves a scatter plot of the 2D embeddings, 
    including metric scores in the title.
    """
    print(f"Generating plot for {dist_type} | nu={nu} | perp={perp}...")
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(Y[:, 0], Y[:, 1], c=labels, cmap='tab10', s=15, alpha=0.8)
    
    cbar = plt.colorbar(scatter, ticks=range(10), label="Digit Class")
    cbar.ax.set_yticklabels([str(i) for i in range(10)]) 
    
    # Construct the highly detailed title
    if tw_score == 0.0 and sil_score == 0.0:
        title_str = (f"Dist: {dist_type} | nu: {nu} | Perp: {perp}\n"
                     f"--- SNAPSHOT ---")
    else:
        title_str = (f"Dist: {dist_type} | nu: {nu} | Perp: {perp}\n"
                     f"TW: {tw_score:.3f} | SIL: {sil_score:.3f}")
    
   
    plt.title(title_str)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close() # Close figure to free up memory during heavy loops
    print(f"Plot saved successfully to: {filename}")