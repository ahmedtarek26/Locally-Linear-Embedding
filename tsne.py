"""
t-SNE - t-Distributed Stochastic Neighbor Embedding

Modern nonlinear method for visualization.
Preserves local neighborhood structure using probability distributions.

Reference: van der Maaten & Hinton (2008)
From course: Modern visualization standard
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE as SklearnTSNE
import pandas as pd

from data_loader import load_swiss_roll, load_faces, load_newsgroups


def apply_tsne(X, perplexity=30, n_components=2):
    """
    Apply t-SNE to data X

    t-SNE minimizes KL divergence between:
    - High-D probability distribution (Gaussian)
    - Low-D probability distribution (Student-t)
    """
    print(f"  Running t-SNE (perplexity={perplexity})...")
    tsne = SklearnTSNE(n_components=n_components, perplexity=perplexity,
                       random_state=42, n_iter=1000)
    Y = tsne.fit_transform(X)
    kl_divergence = tsne.kl_divergence_

    return Y, kl_divergence, tsne


def visualize_tsne(X, Y, y, dataset_name, perplexity, kl_div, results_dir):
    """Create visualization for t-SNE results"""

    if X.shape[1] == 3:
        # Swiss Roll: 3D original + 2D t-SNE
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

        # Right: t-SNE 2D
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.scatter(Y[:, 0], Y[:, 1],
                   c=y, cmap='Spectral', s=20, alpha=0.7)
        ax2.set_title(f'{dataset_name} - t-SNE 2D (perp={perplexity})', 
                     fontsize=13, fontweight='bold')
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.grid(True, alpha=0.3)
    else:
        # High-D data: Just t-SNE 2D
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(Y[:, 0], Y[:, 1],
                  c=y, cmap='Spectral', s=30, alpha=0.7)
        ax.set_title(f'{dataset_name} - t-SNE 2D (perp={perplexity})', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'{dataset_name}: t-SNE (Probabilistic Neighborhood)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()

    plot_path = results_dir / f'tsne_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path


def run_tsne_on_datasets(perplexity=30, results_dir='results'):
    """Apply t-SNE on all datasets"""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)

    print("="*80)
    print("t-SNE - t-DISTRIBUTED STOCHASTIC NEIGHBOR EMBEDDING")
    print("="*80)
    print("Modern method: Preserves local neighborhoods via probabilities")
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

        # Apply t-SNE
        Y, kl_div, tsne = apply_tsne(X, perplexity=perplexity, n_components=2)
        print(f"  ✓ t-SNE complete")
        print(f"  ✓ KL divergence: {kl_div:.6f}")

        # Save embedding
        df = pd.DataFrame(Y, columns=['tSNE1', 'tSNE2'])
        df['label'] = y
        csv_path = results_dir / f'tsne_{dataset_name.lower().replace(" ", "_")}.csv'
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Saved: {csv_path.name}")

        # Visualize
        plot_path = visualize_tsne(X, Y, y, dataset_name, perplexity, kl_div, results_dir)
        print(f"  ✓ Saved: {plot_path.name}")
        print()

        results.append({
            'dataset': dataset_name,
            'method': 't-SNE',
            'perplexity': perplexity,
            'kl_divergence': kl_div
        })

    print("="*80)
    print("✅ t-SNE COMPLETE ON ALL DATASETS")
    print("="*80)

    return results


if __name__ == '__main__':
    run_tsne_on_datasets()