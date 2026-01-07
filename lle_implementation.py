"""
Locally Linear Embedding (LLE) - From Scratch Implementation

Based on: Roweis, S. T., & Saul, L. K. (2000). 
Nonlinear dimensionality reduction by locally linear embedding. 
Science, 290(5500), 2323-2326.

Author: Unsupervised Learning Course Project 2025
University of Trieste
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors


class LocallyLinearEmbedding:
    """
    Locally Linear Embedding for nonlinear dimensionality reduction

    LLE preserves local geometric relationships by:
    1. Finding K nearest neighbors for each point
    2. Computing reconstruction weights that express each point as 
       a linear combination of its neighbors
    3. Finding low-dimensional embeddings that preserve these weights

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the embedded space

    n_neighbors : int, default=10
        Number of nearest neighbors to use for reconstruction
        Note: n_components should be < n_neighbors < n_samples

    reg : float, default=1e-3
        Regularization parameter for conditioning the covariance matrix
        Prevents singularity when n_neighbors > input dimensionality

    Attributes
    ----------
    embedding_ : array, shape (n_samples, n_components)
        The low-dimensional embedding coordinates

    reconstruction_error_ : float
        Sum of squared reconstruction errors in the input space

    Examples
    --------
    >>> from sklearn.datasets import make_swiss_roll
    >>> X, color = make_swiss_roll(n_samples=2000)
    >>> lle = LocallyLinearEmbedding(n_components=2, n_neighbors=12)
    >>> Y = lle.fit_transform(X)
    >>> print(f"Reduced from {X.shape[1]}D to {Y.shape[1]}D")
    """

    def __init__(self, n_components=2, n_neighbors=10, reg=1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.embedding_ = None
        self.reconstruction_error_ = None

    def _compute_reconstruction_weights(self, X, neighbors):
        """
        Step 2: Compute reconstruction weights W for each point

        For each point X_i, solve:
            minimize ||X_i - sum_j W_ij X_j||^2
            subject to: sum_j W_ij = 1, W_ij = 0 if j not neighbor

        Solution uses local covariance matrix C_jk = (X_j - X_i)·(X_k - X_i)
        Weights: w = C^(-1) 1 / (1^T C^(-1) 1)
        """
        N, D = X.shape
        K = self.n_neighbors
        W = np.zeros((N, N))

        for i in range(N):
            neighbor_indices = neighbors[i]

            # Center neighbors around X_i: Z_j = X_j - X_i
            Z = X[neighbor_indices] - X[i]

            # Local covariance: C = Z Z^T
            C = Z @ Z.T

            # Regularize to handle rank deficiency (when K > D)
            C += self.reg * np.trace(C) * np.eye(K)

            # Solve C w = 1 for weights
            w = np.linalg.solve(C, np.ones(K))

            # Normalize to satisfy sum-to-one constraint
            w = w / np.sum(w)

            # Store in weight matrix
            W[i, neighbor_indices] = w

        return W

    def _compute_embedding(self, W):
        """
        Step 3: Compute low-dimensional embedding Y

        Find Y that minimizes:
            Φ(Y) = sum_i ||Y_i - sum_j W_ij Y_j||^2

        This is solved via eigenvalue decomposition of:
            M = (I - W)^T (I - W)

        Solution: bottom (d+1) eigenvectors of M, discarding the first
        (which corresponds to translation mode with eigenvalue ≈ 0)
        """
        N = W.shape[0]

        # M = (I - W)^T (I - W)
        I = np.eye(N)
        M = (I - W).T @ (I - W)

        # Find bottom (d+1) eigenvectors
        eigenvalues, eigenvectors = eigh(
            M, 
            subset_by_index=[0, self.n_components]
        )

        # Discard first eigenvector (translation mode)
        # Keep next d eigenvectors as embedding coordinates
        Y = eigenvectors[:, 1:self.n_components+1]

        return Y

    def fit_transform(self, X):
        """
        Fit LLE model and return embedded coordinates

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data

        Returns
        -------
        Y : array, shape (n_samples, n_components)
            Embedded coordinates
        """
        N, D = X.shape

        # Validate parameters
        if self.n_neighbors >= N:
            raise ValueError(
                f"n_neighbors ({self.n_neighbors}) must be < n_samples ({N})"
            )

        if self.n_components >= self.n_neighbors:
            print(
                f"Warning: n_components ({self.n_components}) should be "
                f"< n_neighbors ({self.n_neighbors}) for best results"
            )

        # Step 1: Find K nearest neighbors
        print(f"Step 1: Finding {self.n_neighbors} nearest neighbors...")
        nbrs = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,
            algorithm='auto'
        )
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)

        # Exclude the point itself (first column)
        neighbors = indices[:, 1:]

        # Step 2: Compute reconstruction weights
        print("Step 2: Computing reconstruction weights...")
        W = self._compute_reconstruction_weights(X, neighbors)

        # Compute reconstruction error
        reconstruction = X - (W @ X)
        self.reconstruction_error_ = np.sum(reconstruction ** 2)
        print(f"   Reconstruction error: {self.reconstruction_error_:.6f}")

        # Step 3: Compute embedding
        print("Step 3: Computing embedding via eigenvalue problem...")
        self.embedding_ = self._compute_embedding(W)

        print(f"✓ LLE completed: {N} points, {D}D → {self.n_components}D")

        return self.embedding_
