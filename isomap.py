"""
Isomap - Isometric Feature Mapping

Nonlinear method that preserves geodesic distances (distances along manifold).
Main comparison algorithm to LLE in the original paper.

Reference: Tenenbaum et al. (2000)
From paper: "Many virtues of LLE are shared by Isomap"
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import Isomap as SklearnIsomap
import pandas as pd

from data_loader import load_swiss_roll, load_faces, load_newsgroups


def apply_isomap(X, n_neighbors=20, n_components=2):
    """
    Apply Isomap to data X

    Isomap:
    1. Build neighborhood graph
    2. Compute shortest paths (geodesic distances)
    3. Apply MDS on geodesic distance matrix
    """
    print(f"  Running Isomap (K={n_neighbors})...")
    isomap = SklearnIsomap(n_neighbors=n_neighbors, n_components=n_components)
    Y = isomap.fit_transform(X)
    reconstruction_error = isomap.reconstruction_error()

    return Y, reconstruction_error, isomap


def visualize_isomap(X, Y, y, dataset_name, k, error, results_dir):
    """Create visualization for Isomap results"""

    if X.shape[1] == 3:
        # Swiss Roll: 3D original + 2D Isomap
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

        # Right: Isomap 2D
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(Y[:, 0], Y[:, 1],
                   c=y, cmap='Spectral', s=20, alpha=0.7)
        ax2.set_title(f'{dataset_name} - Isomap 2D (K={k})', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('Isomap 1')
        ax2.set_ylabel('Isomap 2')
        ax2.grid(True, alpha=0.3)
    else:
        # High-D data: Just Isomap 2D
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(Y[:, 0], Y[:, 1],
                  c=y, cmap='Spectral', s=30, alpha=0.7)
        ax.set_title(f'{dataset_name} - Isomap 2D (K={k})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Isomap 1', fontsize=12)
        ax.set_ylabel('Isomap 2', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{dataset_name}: Isomap (Geodesic Distance Preservation)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot_path = results_dir / f'isomap_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def run_isomap_on_datasets(k_swiss=20, k_faces=12, k_news=10, results_dir='results'):
    """Apply Isomap on all datasets"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    print("="*80)
    print("ISOMAP - ISOMETRIC FEATURE MAPPING")
    print("="*80)
    print("Nonlinear method: Preserves geodesic (manifold) distances")
    print("Main comparison to LLE from the paper")
    print("="*80)
    print()

    # Load datasets
    datasets = [
        ('Swiss Roll', *load_swiss_roll(n_samples=1000), k_swiss),
        ('Faces', *load_faces(n_samples=400), k_faces),
        ('Newsgroups', *load_newsgroups(n_samples=1000), k_news)
    ]

    results = []

    for dataset_name, X, y, info, k in datasets:
        print(f"Processing: {dataset_name}")
        print(f"  Shape: {X.shape}, K={k}")

        # Apply Isomap
        Y, error, isomap = apply_isomap(X, n_neighbors=k, n_components=2)
        print(f"  ✓ Isomap complete")
        print(f"  ✓ Reconstruction error: {error:.6f}")

        # Save embedding
        df = pd.DataFrame(Y, columns=['Isomap1', 'Isomap2'])
        df['label'] = y
        csv_path = results_dir / f'isomap_{dataset_name.lower().replace(" ", "_")}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path.name}")

        # Visualize
        plot_path = visualize_isomap(X, Y, y, dataset_name, k, error, results_dir)
        print(f"  ✓ Saved: {plot_path.name}")
        print()

        results.append({
            'dataset': dataset_name,
            'method': 'Isomap',
            'K': k,
            'reconstruction_error': error
        })

    print("="*80)
    print("✅ ISOMAP COMPLETE ON ALL DATASETS")
    print("="*80)

    return results


if __name__ == '__main__':
    run_isomap_on_datasets()