"""
MDS - Multidimensional Scaling (Classical)

Preserves pairwise distances between points.
Classical MDS is equivalent to PCA when using Euclidean distances.

Reference: Cox & Cox (1994)
From paper: Compared with LLE as traditional method
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import MDS as SklearnMDS
import pandas as pd

from data_loader import load_swiss_roll, load_faces, load_newsgroups


def apply_mds(X, n_components=2):
    """Apply Classical MDS to data X"""
    print("  Running MDS (this may take a while)...")
    mds = SklearnMDS(n_components=n_components, random_state=42, 
                     max_iter=300, n_init=1)
    Y = mds.fit_transform(X)
    stress = mds.stress_

    return Y, stress, mds


def visualize_mds(X, Y, y, dataset_name, stress, results_dir):
    """Create visualization for MDS results"""

    if X.shape[1] == 3:
        # Swiss Roll: 3D original + 2D MDS
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

        # Right: MDS 2D
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(Y[:, 0], Y[:, 1],
                   c=y, cmap='Spectral', s=20, alpha=0.7)
        ax2.set_title(f'{dataset_name} - MDS 2D (stress={stress:.2f})', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('MDS 1')
        ax2.set_ylabel('MDS 2')
        ax2.grid(True, alpha=0.3)
    else:
        # High-D data: Just MDS 2D
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(Y[:, 0], Y[:, 1],
                  c=y, cmap='Spectral', s=30, alpha=0.7)
        ax.set_title(f'{dataset_name} - MDS 2D (stress={stress:.2f})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('MDS 1', fontsize=12)
        ax.set_ylabel('MDS 2', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{dataset_name}: MDS (Distance Preservation)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot_path = results_dir / f'mds_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def run_mds_on_datasets(results_dir='results'):
    """Apply MDS on all datasets"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    print("="*80)
    print("MDS - MULTIDIMENSIONAL SCALING")
    print("="*80)
    print("Classical method: Preserves pairwise distances")
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

        # Apply MDS
        Y, stress, mds = apply_mds(X, n_components=2)
        print(f"  ✓ MDS complete")
        print(f"  ✓ Stress: {stress:.2f}")

        # Save embedding
        df = pd.DataFrame(Y, columns=['MDS1', 'MDS2'])
        df['label'] = y
        csv_path = results_dir / f'mds_{dataset_name.lower().replace(" ", "_")}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path.name}")

        # Visualize
        plot_path = visualize_mds(X, Y, y, dataset_name, stress, results_dir)
        print(f"  ✓ Saved: {plot_path.name}")
        print()

        results.append({
            'dataset': dataset_name,
            'method': 'MDS',
            'stress': stress
        })

    print("="*80)
    print("✅ MDS COMPLETE ON ALL DATASETS")
    print("="*80)

    return results


if __name__ == '__main__':
    run_mds_on_datasets()