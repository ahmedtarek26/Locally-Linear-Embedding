"""
Course Lab Methods - UL25 Lab Implementations

Wrappers for dimensionality reduction methods following the 
exact style and approach from Unsupervised Learning course labs.

Course: Unsupervised Learning 2025
University of Trieste
Professor: Alejandro Rodriguez Garcia
Lab Instructor: Francesco Tomba
GitHub: https://github.com/lykos98/UL25
"""

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, TSNE, MDS
import umap


class CourseLabMethods:
    """
    Dimensionality reduction methods from course labs

    These mirror the implementations from:
    - Lab 2 (Lab2-PCA.ipynb): PCA, MDS
    - Lab 3 (Lab3-Isomap.ipynb): Isomap
    - Lab 4 (Lab4-KernelPCA.ipynb): Kernel PCA
    - Lab 5 (Lab5-DimensionalityReduction.ipynb): t-SNE, UMAP
    """

    @staticmethod
    def lab2_pca(X, n_components=2):
        """
        PCA following Lab 2 (Lab2-PCA.ipynb)

        Principal Component Analysis:
        - Center data (subtract mean)
        - Compute covariance matrix
        - Find eigenvectors (principal components)
        - Project data onto top components

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        n_components : int
            Number of components to keep

        Returns
        -------
        dict with keys:
            - embedding: low-dimensional coordinates
            - model: fitted PCA object
            - explained_variance: total variance explained
            - lab: source lab
            - type: method type
        """
        pca = PCA(n_components=n_components)
        Y = pca.fit_transform(X)

        return {
            'embedding': Y,
            'model': pca,
            'explained_variance': pca.explained_variance_ratio_.sum(),
            'lab': 'Lab 2',
            'type': 'linear'
        }

    @staticmethod
    def lab2_mds(X, n_components=2):
        """
        Classical MDS following Lab 2

        Multidimensional Scaling:
        - Compute pairwise Euclidean distances
        - Apply classical MDS (eigenvalue decomposition)
        - For Euclidean distance, equivalent to PCA

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        n_components : int
            Number of dimensions in output

        Returns
        -------
        dict with embedding and metadata
        """
        mds = MDS(n_components=n_components, metric=True, 
                  dissimilarity='euclidean', random_state=42, 
                  n_init=1, max_iter=300)
        Y = mds.fit_transform(X)

        return {
            'embedding': Y,
            'model': mds,
            'stress': mds.stress_,
            'lab': 'Lab 2',
            'type': 'distance_based'
        }

    @staticmethod
    def lab3_isomap(X, n_components=2, n_neighbors=12):
        """
        Isomap following Lab 3 (Lab3-Isomap.ipynb)

        Isometric Feature Mapping:
        1. Build K-nearest neighbor graph
        2. Compute geodesic distances (shortest paths)
        3. Apply classical MDS on geodesic distances

        Key difference from LLE: uses global geodesic distances,
        not local linear reconstructions

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        n_components : int
            Number of output dimensions
        n_neighbors : int
            Number of neighbors for graph construction

        Returns
        -------
        dict with embedding and metadata
        """
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        Y = isomap.fit_transform(X)

        return {
            'embedding': Y,
            'model': isomap,
            'reconstruction_error': isomap.reconstruction_error(),
            'lab': 'Lab 3',
            'type': 'nonlinear_global',
            'params': {'n_neighbors': n_neighbors}
        }

    @staticmethod
    def lab4_kernel_pca(X, n_components=2, kernel='rbf', gamma=None):
        """
        Kernel PCA following Lab 4 (Lab4-KernelPCA.ipynb)

        Kernel Principal Component Analysis:
        - Map data to high-dimensional feature space via kernel
        - Perform PCA in that space
        - RBF kernel: k(x,y) = exp(-γ||x-y||²)

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        n_components : int
            Number of components to extract
        kernel : str
            Kernel type ('rbf', 'poly', 'linear')
        gamma : float, optional
            RBF kernel parameter

        Returns
        -------
        dict with embedding and metadata
        """
        kpca = KernelPCA(n_components=n_components, kernel=kernel, 
                         gamma=gamma, random_state=42)
        Y = kpca.fit_transform(X)

        return {
            'embedding': Y,
            'model': kpca,
            'lab': 'Lab 4',
            'type': 'nonlinear_kernel',
            'params': {'kernel': kernel}
        }

    @staticmethod
    def lab5_tsne(X, n_components=2, perplexity=30, random_state=42):
        """
        t-SNE following Lab 5 (Lab5-DimensionalityReduction.ipynb)

        t-distributed Stochastic Neighbor Embedding:
        - Model pairwise similarities with Gaussian in high-D
        - Model pairwise similarities with t-distribution in low-D
        - Minimize KL divergence via gradient descent
        - Perplexity ≈ effective number of neighbors

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        n_components : int
            Number of output dimensions (typically 2 or 3)
        perplexity : float
            Related to number of nearest neighbors
        random_state : int
            Random seed for reproducibility

        Returns
        -------
        dict with embedding and metadata
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity,
                    random_state=random_state, max_iter=1000)
        Y = tsne.fit_transform(X)

        return {
            'embedding': Y,
            'model': tsne,
            'kl_divergence': tsne.kl_divergence_,
            'lab': 'Lab 5',
            'type': 'nonlinear_probabilistic',
            'params': {'perplexity': perplexity}
        }

    @staticmethod
    def lab5_umap(X, n_components=2, n_neighbors=15, random_state=42):
        """
        UMAP following Lab 5 (Lab5-DimensionalityReduction.ipynb)

        Uniform Manifold Approximation and Projection:
        - Build fuzzy topological representation of data
        - Optimize low-dimensional layout
        - Balances local and global structure
        - Generally faster than t-SNE

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        n_components : int
            Number of output dimensions
        n_neighbors : int
            Number of neighbors for manifold approximation
        random_state : int
            Random seed

        Returns
        -------
        dict with embedding and metadata
        """
        umap_model = umap.UMAP(n_components=n_components, 
                               n_neighbors=n_neighbors, 
                               random_state=random_state)
        Y = umap_model.fit_transform(X)

        return {
            'embedding': Y,
            'model': umap_model,
            'lab': 'Lab 5',
            'type': 'nonlinear_topological',
            'params': {'n_neighbors': n_neighbors}
        }
