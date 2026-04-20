#
#  tsne.py
#
# Implementation of t-SNE using a custom Autograd engine for flexible
# heavy-tailed distributions.
#

import numpy as np
import pylab
from autograd import Value # Import your custom engine
import gc


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    for i in range(n):
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

# --- MODIFIED: Added dist_type to the signature for modular routing ---
def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, nu=1.0, dist_type='t-distribution', snapshot_iters=None):
    """
        Runs parameterized t-SNE on the dataset. 
        `nu` is the degrees of freedom for the heavy-tailed distribution.
        nu = 1.0 corresponds to standard t-SNE (Cauchy distribution).
    """

    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    # --- MODIFIED: Added check to skip PCA if the orchestrator already did it ---
    if X.shape[1] > initial_dims:
        X = pca(X, initial_dims).real # X is now a PCA-reduced matrix (n x 50)
    else:
        print("Data is already PCA-reduced. Skipping internal PCA step.")
    # ----------------------------------------------------------------------------
    
    (n, d) = X.shape
    max_iter = 500
    #based on observing convergence i am doing an early stopping to 500 iterations but initially i kept it at 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    
    # ==============================================================================
    # THE INITIALIZATION CHANGE (Random vs. PCA)
    # ==============================================================================
    # BASELINE: 
    # Y = np.random.randn(n, no_dims) 
    # The baseline scatters points using random Gaussian noise. This can lead to 
    # slow convergence, getting trapped in bad local minima, and causes the 
    # final map to be rotated/flipped randomly every time you run the script.
    #
    # WHAT WE DID: 
    # Replaced random noise with the first two Principal Components of the dataset.
    # Because X is already the PCA-reduced matrix, X[:, :no_dims] extracts the 
    # top 2 components. We scale it down significantly (0.0001) to prevent the 
    # massive attractive forces during early exaggeration from causing overflow.
    #
    # WHY WE DID IT:
    # This acts as a "rough draft" for the global structure. It ensures that 
    # globally separated clusters start on opposite sides of the map, preventing 
    # them from having to pass through each other. It guarantees determinism 
    # (the map orientation will look the same every run) and improves stability.
    # ==============================================================================
    
    # Y is kept as a NumPy array for the momentum updates
    Y = X[:, :no_dims].copy()
    Y = (Y / np.std(Y[:, 0])) * 0.0001
    
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.#early exaggeration default is 4 please change it back to 4 if it is not 4..we need to divide by 4 later 
    P = np.maximum(P, 1e-12)

    # Autograd mask to zero out the diagonal
    diag_mask = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(diag_mask, 0.0)

    print(f"\n--- Starting Autograd Descent [{dist_type} | nu={nu} | Perp={perplexity}] ---")

    # Run iterations
    for iter in range(max_iter):

        # -------------------------------------------------------------
        # AUTOGRAD FORWARD PASS
        # -------------------------------------------------------------
        Y_val = Value(Y) # Wrap the current NumPy positions in autograd
        
        # 1. Pairwise squared distances: D = sum(Y^2) + sum(Y^2).T - 2*(Y @ Y.T)
        Y_sq = Y_val ** 2
        sum_Y = Y_sq.sum(axis=1, keepdims=True)
        D = sum_Y + sum_Y.T - (Y_val @ Y_val.T) * 2.0
        
        # -------------------------------------------------------------
        # --- MODIFIED: Distribution Router for Testbed ---
        # -------------------------------------------------------------
        if dist_type == 't-distribution':
            # Parameterized Student's t-distribution
            exponent = -(nu + 1.0) / 2.0
            num = (D * (1.0 / nu) + 1.0) ** exponent
            
        elif dist_type == 'power-law':
            # Decoupled Inverse Power-Law (Extreme heavy tails)
            num = (D + 1.0) ** -nu
            
        elif dist_type == 'gaussian':
            # Standard SNE (Light tails, causes crowding)
            num = (-D).exp()
            
        elif dist_type == 'logistic':
            # Logistic SNE (Math failure state / vanishing gradients)
            num = (D.exp() + 1.0) ** -1.0
            
        else:
            raise ValueError(f"Unknown dist_type: {dist_type}. Check orchestrator script.")
        # -------------------------------------------------------------
        
        # 3. Mask out the diagonal 
        num_masked = num * diag_mask
        
        # 4. Normalize to get Q
        Q_val = num_masked / num_masked.sum()
        
        # 5. Calculate Loss (Cross Entropy equivalent of KL Divergence for fixed P)
        # Loss = -sum(P * log(Q))
        loss = -(Q_val.log() * P).sum()
        
        # -------------------------------------------------------------
        # AUTOGRAD BACKWARD PASS
        # -------------------------------------------------------------
        loss.backward()
        
        # Extract the computed gradient back into NumPy format
        dY = Y_val.grad

        # -------------------------------------------------------------
        # STANDARD MOMENTUM & WEIGHT UPDATE (Untouched)
        # -------------------------------------------------------------
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
            
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute and print current value of cost function
        if (iter + 1) % 10 == 0:
            # We use NumPy here purely to print the exact KL divergence
            Q_np = np.maximum(Q_val.data, 1e-12)
            C = np.sum(P * np.log(P / Q_np))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop early exaggeration
        if iter == 100:
            P = P / 4.
        # Force Python to flush the RAM
        gc.collect()

    snapshots={}

    # Run iterations
    for iter in range(max_iter):
        
        # ... [All your existing forward/backward/momentum code stays exactly the same here] ...
        
        # ---> Add this at the very end of the loop (inside the loop):
        if snapshot_iters is not None and iter in snapshot_iters:
            snapshots[iter] = Y.copy() # Save a copy of the exact coordinates at this moment
            print(f"📸 Captured snapshot at iteration {iter}")

    # ---> Modify the return statement at the end of the function:
    if snapshot_iters is not None:
        snapshots['final'] = Y.copy()
        return snapshots
    
    return Y


# ==============================================================================
# HOW TO USE THIS MODULE
# ==============================================================================
# This script contains the core parameterized t-SNE algorithm and should NOT 
# be run directly. Import it into your main execution scripts.
#
# Example Usage:
# --------------
# import numpy as np
# import matplotlib.pyplot as plt
# from tsne import tsne
#
# # 1. Load your high-dimensional data (N x D numpy array)
# X_high_dim = np.loadtxt("my_data.txt") 
#
# # 2. Run the algorithm to get the (N x 2) low-dimensional embeddings
# #    nu=1.0 is standard t-SNE. Lower nu for heavier tails.
# Y_low_dim = tsne(X_high_dim, no_dims=2, initial_dims=50, perplexity=30.0, nu=0.5, dist_type='t-distribution')
#
# # 3. Plot the results
# plt.scatter(Y_low_dim[:, 0], Y_low_dim[:, 1])
# plt.show()
# ==============================================================================