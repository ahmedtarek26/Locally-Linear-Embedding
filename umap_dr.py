"""
UMAP - Uniform Manifold Approximation and Projection

State-of-the-art nonlinear method.
Faster than t-SNE, preserves more global structure.

Reference: McInnes, Healy & Melville (2018)
From course: Modern best practice for large datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

from data_loader import load_swiss_roll, load_faces, load_newsgroups

# Try to import UMAP
try:
    from umap_dr import UMAP as UMAPLib
    UMAP_AVAILABLE = True
except ImportError:
    print("⚠️  UMAP not installed. Install with: pip install umap-learn")
    UMAP_AVAILABLE = False


def apply_umap(X, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Apply UMAP to data X

    UMAP:
    1. Build fuzzy topological representation of high-D data
    2. Find low-D representation with similar structure
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")

    print(f"  Running UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    umap = UMAPLib(n_neighbors=n_neighbors, min_dist=min_dist, 
                   n_components=n_components, random_state=42)
    Y = umap.fit_transform(X)

    return Y, umap


def visualize_umap(X, Y, y, dataset_name, n_neighbors, min_dist, results_dir):
    """Create visualization for UMAP results"""

    if X.shape[1] == 3:
        # Swiss Roll: 3D original + 2D UMAP
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

        # Right: UMAP 2D
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(Y[:, 0], Y[:, 1],
                   c=y, cmap='Spectral', s=20, alpha=0.7)
        ax2.set_title(f'{dataset_name} - UMAP 2D (K={n_neighbors})', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('UMAP 1')
        ax2.set_ylabel('UMAP 2')
        ax2.grid(True, alpha=0.3)
    else:
        # High-D data: Just UMAP 2D
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(Y[:, 0], Y[:, 1],
                  c=y, cmap='Spectral', s=30, alpha=0.7)
        ax.set_title(f'{dataset_name} - UMAP 2D (K={n_neighbors})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1', fontsize=12)
        ax.set_ylabel('UMAP 2', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{dataset_name}: UMAP (Topological Structure)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot_path = results_dir / f'umap_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def run_umap_on_datasets(n_neighbors=15, min_dist=0.1, results_dir='results'):
    """Apply UMAP on all datasets"""
    if not UMAP_AVAILABLE:
        print("="*80)
        print("⚠️  UMAP NOT AVAILABLE")
        print("="*80)
        print("Install with: pip install umap-learn")
        print("="*80)
        return []

    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    print("="*80)
    print("UMAP - UNIFORM MANIFOLD APPROXIMATION AND PROJECTION")
    print("="*80)
    print("State-of-the-art: Fast, preserves local+global structure")
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

        try:
            # Apply UMAP
            Y, umap = apply_umap(X, n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
            print(f"  ✓ UMAP complete")

            # Save embedding
            df = pd.DataFrame(Y, columns=['UMAP1', 'UMAP2'])
            df['label'] = y
            csv_path = results_dir / f'umap_{dataset_name.lower().replace(" ", "_")}.csv'
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved: {csv_path.name}")

            # Visualize
            plot_path = visualize_umap(X, Y, y, dataset_name, n_neighbors, min_dist, results_dir)
            print(f"  ✓ Saved: {plot_path.name}")
            print()

            results.append({
                'dataset': dataset_name,
                'method': 'UMAP',
                'n_neighbors': n_neighbors,
                'min_dist': min_dist
            })
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print()

    print("="*80)
    print("✅ UMAP COMPLETE ON ALL DATASETS")
    print("="*80)

    return results


if __name__ == '__main__':
    run_umap_on_datasets()