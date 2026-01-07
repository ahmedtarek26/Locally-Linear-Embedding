"""
Locally Linear Embedding (LLE) - Complete Implementation

Based on: Roweis, S. T., & Saul, L. K. (2000). 
Nonlinear dimensionality reduction by locally linear embedding.
Science, 290(5500), 2323-2326.

This file contains:
1. LLE algorithm implementation
2. K selection experiment
3. LLE application on all datasets (including robust handling for sparse data)
4. Visualization and results saving
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.linalg import eigh
from sklearn.neighbors import NearestNeighbors
import pandas as pd

from data_loader import load_swiss_roll, load_faces, load_newsgroups


# ==============================================================================
# PART 1: LLE ALGORITHM IMPLEMENTATION
# ==============================================================================

class LLE:
    """
    Locally Linear Embedding

    Parameters
    ----------
    n_components : int, default=2
        Number of dimensions in the embedding space

    n_neighbors : int, default=12
        Number of neighbors for each point (K in the paper)

    reg : float, default=1e-3
        Regularization parameter

    robust : bool, default=False
        Use robust weight computation for sparse/difficult data
    """

    def __init__(self, n_components=2, n_neighbors=12, reg=1e-3, robust=False):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg
        self.robust = robust
        self.embedding_ = None
        self.reconstruction_error_ = None
        self.weights_ = None

    def fit_transform(self, X):
        """
        Fit LLE model to X and return embedding
        """
        N, D = X.shape

        if self.n_neighbors >= N:
            raise ValueError(f"n_neighbors ({self.n_neighbors}) must be < n_samples ({N})")

        print(f"\nLLE: Reducing {N} points from {D}D to {self.n_components}D")
        print(f"Parameters: K={self.n_neighbors}, reg={self.reg}, robust={self.robust}")
        print("-" * 60)

        # Step 1: Find neighbors
        print("Step 1/3: Finding K nearest neighbors...")
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors + 1, algorithm='auto')
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        neighbors = indices[:, 1:]
        print(f"  ✓ Found {self.n_neighbors} neighbors for each point")

        # Step 2: Compute weights
        if self.robust:
            print("Step 2/3: Computing reconstruction weights (ROBUST mode)...")
        else:
            print("Step 2/3: Computing reconstruction weights...")

        self.weights_ = np.zeros((N, N))

        for i in range(N):
            neighbor_indices = neighbors[i]
            Z = X[neighbor_indices] - X[i]
            C = Z @ Z.T

            if self.robust:
                # Robust: Use stronger regularization
                trace_C = max(np.trace(C), 1.0)
                C += self.reg * trace_C * np.eye(self.n_neighbors)

                try:
                    w = np.linalg.solve(C, np.ones(self.n_neighbors))
                except:
                    # Extra regularization if needed
                    C += 10 * self.reg * trace_C * np.eye(self.n_neighbors)
                    try:
                        w = np.linalg.solve(C, np.ones(self.n_neighbors))
                    except:
                        # Last resort: uniform weights
                        w = np.ones(self.n_neighbors) / self.n_neighbors

                w = w / (np.sum(w) + 1e-10)
            else:
                # Standard: Regular regularization
                C += self.reg * np.trace(C) * np.eye(self.n_neighbors)
                w = np.linalg.solve(C, np.ones(self.n_neighbors))
                w = w / np.sum(w)

            self.weights_[i, neighbor_indices] = w

        reconstruction = X - (self.weights_ @ X)
        self.reconstruction_error_ = np.sum(reconstruction ** 2)
        print(f"  ✓ Weights computed, reconstruction error: {self.reconstruction_error_:.6f}")

        # Step 3: Compute embedding
        print("Step 3/3: Computing embedding (eigenvalue problem)...")
        I = np.eye(N)
        M = (I - self.weights_).T @ (I - self.weights_)

        # Ensure symmetry for numerical stability
        M = (M + M.T) / 2

        eigenvalues, eigenvectors = eigh(M, subset_by_index=[0, self.n_components])
        self.embedding_ = eigenvectors[:, 1:self.n_components + 1]
        print(f"  ✓ Embedding computed")

        print("-" * 60)
        print(f"✓ LLE complete: {N} points, {D}D → {self.n_components}D")

        return self.embedding_



# ==============================================================================
# PART 2: APPLY LLE ON ALL DATASETS
# ==============================================================================

def apply_lle_on_datasets(k_swiss=20, k_faces=12, k_news=10, results_dir='results'):
    """
    Apply LLE on all datasets with chosen K values
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    print("="*80)
    print("APPLYING LLE ON ALL DATASETS")
    print("="*80)
    print()

    # Load datasets
    print("Loading datasets...")
    X_swiss, y_swiss, _ = load_swiss_roll(n_samples=1000)
    X_faces, y_faces, _ = load_faces(n_samples=400)
    X_news, y_news, _ = load_newsgroups(n_samples=1000)
    print()

    # Process Swiss Roll
    print("="*80)
    print("1. SWISS ROLL")
    print("="*80)
    lle_swiss = LLE(n_components=2, n_neighbors=k_swiss, reg=1e-3)
    Y_swiss = lle_swiss.fit_transform(X_swiss)

    # Save embedding
    df = pd.DataFrame(Y_swiss, columns=['component_1', 'component_2'])
    df['label'] = y_swiss
    df.to_csv(results_dir / 'embedding_swiss_roll.csv', index=False)
    print(f"✓ Saved: embedding_swiss_roll.csv")

    # Visualize: 3D rotated + LLE 2D (NO COLORBAR)
    fig = plt.figure(figsize=(14, 6))

    # Left: 3D rotated view
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2],
               c=y_swiss, cmap='Spectral', s=20, alpha=0.7)
    ax1.set_title('Swiss Roll - Original 3D', fontsize=13, fontweight='bold')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=10, azim=45)

    # Right: LLE 2D
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(Y_swiss[:, 0], Y_swiss[:, 1],
               c=y_swiss, cmap='Spectral', s=20, alpha=0.7)
    ax2.set_title(f'Swiss Roll - LLE 2D (K={k_swiss})', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Swiss Roll: Before and After LLE', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(results_dir / 'lle_swiss_roll.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: lle_swiss_roll.png")
    print()

    # Process Faces
    print("="*80)
    print("2. FACES")
    print("="*80)
    lle_faces = LLE(n_components=2, n_neighbors=k_faces, reg=1e-3)
    Y_faces = lle_faces.fit_transform(X_faces)

    # Save embedding
    df = pd.DataFrame(Y_faces, columns=['component_1', 'component_2'])
    df['label'] = y_faces
    df.to_csv(results_dir / 'embedding_faces.csv', index=False)
    print(f"✓ Saved: embedding_faces.csv")

    # Visualize: LLE 2D only (NO COLORBAR)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(Y_faces[:, 0], Y_faces[:, 1],
              c=y_faces, cmap='Spectral', s=30, alpha=0.7)
    ax.set_title(f'Faces - LLE 2D Embedding (K={k_faces})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'lle_faces.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: lle_faces.png")
    print()

    # Process Newsgroups (ROBUST MODE)
    print("="*80)
    print("3. NEWSGROUPS")
    print("="*80)
    try:
        lle_news = LLE(n_components=2, n_neighbors=k_news, reg=10.0, robust=True)
        Y_news = lle_news.fit_transform(X_news)

        # Save embedding
        df = pd.DataFrame(Y_news, columns=['component_1', 'component_2'])
        df['label'] = y_news
        df.to_csv(results_dir / 'embedding_newsgroups.csv', index=False)
        print(f"✓ Saved: embedding_newsgroups.csv")

        # Visualize: LLE 2D only (NO COLORBAR)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(Y_news[:, 0], Y_news[:, 1],
                  c=y_news, cmap='Spectral', s=30, alpha=0.7)
        ax.set_title(f'Newsgroups - LLE 2D Embedding (K={k_news})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Component 1', fontsize=12)
        ax.set_ylabel('Component 2', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(results_dir / 'lle_newsgroups.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: lle_newsgroups.png")
        print()
    except Exception as e:
        print(f"✗ Newsgroups failed: {e}")
        print()

    print("="*80)
    print("✅ ALL DATASETS PROCESSED")
    print("="*80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'apply':
        # Apply LLE only
        apply_lle_on_datasets()
    else:
        # Run everything
        print("\n" + "="*80)
        print("COMPLETE LLE WORKFLOW")
        print("="*80)
        print()
        print("Step 1: Apply LLE")
        apply_lle_on_datasets()
        print()
        print("="*80)
        print("✅ COMPLETE!")
        print("="*80)