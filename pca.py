"""
PCA - Principal Component Analysis

Linear dimensionality reduction using SVD.
Finds directions of maximum variance.

Reference: Pearson (1901)
From course: Linear baseline for comparison with nonlinear methods
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from data_loader import load_swiss_roll, load_faces, load_newsgroups


def apply_pca(X, n_components=2):
    """Apply PCA to data X"""
    pca = SklearnPCA(n_components=n_components)
    Y = pca.fit_transform(X)

    # Get explained variance
    explained_var = pca.explained_variance_ratio_.sum()

    return Y, explained_var, pca


def visualize_pca(X, Y, y, dataset_name, explained_var, results_dir):
    """Create visualization for PCA results"""

    if X.shape[1] == 3:
        # Swiss Roll: 3D original + 2D PCA
        fig = plt.figure(figsize=(14, 6))

        # Left: Original 3D
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(X[:, 0], X[:, 1], X[:, 2],
                   c=y, cmap='Spectral', s=20, alpha=0.7)
        ax1.set_title(f'{dataset_name} - Original 3D', fontsize=13, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.view_init(elev=10, azim=45)

        # Right: PCA 2D
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(Y[:, 0], Y[:, 1],
                   c=y, cmap='Spectral', s=20, alpha=0.7)
        ax2.set_title(f'{dataset_name} - PCA 2D ({explained_var:.1%} var)', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('PC 1')
        ax2.set_ylabel('PC 2')
        ax2.grid(True, alpha=0.3)
    else:
        # High-D data: Just PCA 2D
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(Y[:, 0], Y[:, 1],
                  c=y, cmap='Spectral', s=30, alpha=0.7)
        ax.set_title(f'{dataset_name} - PCA 2D ({explained_var:.1%} var)', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('PC 1', fontsize=12)
        ax.set_ylabel('PC 2', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{dataset_name}: PCA (Linear Method)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot_path = results_dir / f'pca_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def run_pca_on_datasets(results_dir='results'):
    """Apply PCA on all datasets"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    print("="*80)
    print("PCA - PRINCIPAL COMPONENT ANALYSIS")
    print("="*80)
    print("Linear method: Finds directions of maximum variance")
    print("="*80)
    print()

    # Load datasets
    datasets = [
        ('Swiss Roll', *load_swiss_roll(n_samples=1000)),
        ('Faces', *load_faces(n_samples=400)),
        ('Newsgroups', *load_newsgroups(n_samples=1000))
    ]

    results = []

    for dataset_name, X, y, info in datasets:
        print(f"Processing: {dataset_name}")
        print(f"  Shape: {X.shape}")

        # Apply PCA
        Y, explained_var, pca = apply_pca(X, n_components=2)
        print(f"  ✓ PCA complete")
        print(f"  ✓ Explained variance: {explained_var:.2%}")

        # Save embedding
        df = pd.DataFrame(Y, columns=['PC1', 'PC2'])
        df['label'] = y
        csv_path = results_dir / f'pca_{dataset_name.lower().replace(" ", "_")}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path.name}")

        # Visualize
        plot_path = visualize_pca(X, Y, y, dataset_name, explained_var, results_dir)
        print(f"  ✓ Saved: {plot_path.name}")
        print()

        results.append({
            'dataset': dataset_name,
            'method': 'PCA',
            'explained_variance': explained_var
        })

    print("="*80)
    print("✅ PCA COMPLETE ON ALL DATASETS")
    print("="*80)

    return results


if __name__ == '__main__':
    run_pca_on_datasets()