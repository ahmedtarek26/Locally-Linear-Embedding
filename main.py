"""
Main Script - Dimensionality Reduction Comparison

Runs all algorithms on all datasets and creates comparison visualizations.

Algorithms:
- LLE (Locally Linear Embedding) - Your implementation
- PCA (Principal Component Analysis) - Linear baseline
- MDS (Multidimensional Scaling) - Distance preservation
- Isomap - Geodesic distance preservation
- t-SNE - Probabilistic neighborhoods
- UMAP - Topological structure

Usage:
  python main.py              # Run all algorithms
  python main.py --quick      # Run only fast algorithms (LLE, PCA, Isomap)
  python main.py --dataset swiss_roll  # Run on specific dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import time
import argparse

# Import all algorithm modules
import lle
import pca
import mds
import isomap
import tsne
try:
    import umap_dr
    UMAP_AVAILABLE = True
except:
    UMAP_AVAILABLE = False

from data_loader import load_swiss_roll, load_faces, load_newsgroups


def create_comparison_plot(dataset_name, results_dir='results'):
    """
    Create side-by-side comparison of all methods for one dataset
    """
    print(f"\nCreating comparison plot for {dataset_name}...")

    # Load all embeddings
    methods = ['lle', 'pca', 'mds', 'isomap', 'tsne']
    if UMAP_AVAILABLE:
        methods.append('umap')

    embeddings = {}
    for method in methods:
        csv_path = results_dir / f'{method}_{dataset_name.lower().replace(" ", "_")}.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            embeddings[method.upper()] = df

    if not embeddings:
        print(f"  ⚠️  No embeddings found for {dataset_name}")
        return

    # Create subplot grid
    n_methods = len(embeddings)
    ncols = 3
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 6*nrows))
    axes = axes.flatten() if n_methods > 1 else [axes]

    for idx, (method, df) in enumerate(embeddings.items()):
        ax = axes[idx]

        # Get column names (first two are coordinates, last is label)
        cols = df.columns.tolist()
        x_col, y_col = cols[0], cols[1]
        labels = df['label'].values

        # Plot
        scatter = ax.scatter(df[x_col], df[y_col],
                           c=labels, cmap='Spectral', s=20, alpha=0.7)
        ax.set_title(method, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.grid(True, alpha=0.3)

    # Hide extra subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{dataset_name}: All Methods Comparison', 
                fontsize=18, fontweight='bold')
    plt.tight_layout()

    plot_path = results_dir / f'comparison_{dataset_name.lower().replace(" ", "_")}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {plot_path.name}")


def create_summary_table(all_results, results_dir='results'):
    """Create summary table of all results"""
    print("\nCreating summary table...")

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Save as CSV
    csv_path = results_dir / 'summary_all_methods.csv'
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: {csv_path.name}")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    return df


def run_all_algorithms(datasets_to_run=None, quick_mode=False):
    """
    Run all dimensionality reduction algorithms

    Parameters:
    -----------
    datasets_to_run : list or None
        List of dataset names to process. If None, process all.
    quick_mode : bool
        If True, skip slow methods (MDS, t-SNE, UMAP)
    """
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)

    print("="*80)
    print("DIMENSIONALITY REDUCTION - COMPLETE COMPARISON")
    print("="*80)
    print("Based on Roweis & Saul (2000) LLE paper")
    print("="*80)
    print()

    # Determine which datasets to run
    all_datasets = ['Swiss Roll', 'Faces', 'Newsgroups']
    if datasets_to_run:
        datasets = [d for d in all_datasets if d in datasets_to_run]
    else:
        datasets = all_datasets

    print(f"Datasets: {', '.join(datasets)}")
    print(f"Quick mode: {quick_mode}")
    print()

    all_results = []

    # 1. LLE
    print("\n" + "="*80)
    print("1/6: RUNNING LLE")
    print("="*80)
    start = time.time()
    lle.apply_lle_on_datasets(results_dir=results_dir)
    elapsed = time.time() - start
    all_results.append({'method': 'LLE', 'time': elapsed})

    # 2. PCA
    print("\n" + "="*80)
    print("2/6: RUNNING PCA")
    print("="*80)
    start = time.time()
    pca.run_pca_on_datasets(results_dir=results_dir)
    elapsed = time.time() - start
    all_results.append({'method': 'PCA', 'time': elapsed})

    if not quick_mode:
        # 3. MDS
        print("\n" + "="*80)
        print("3/6: RUNNING MDS (SLOW)")
        print("="*80)
        start = time.time()
        mds.run_mds_on_datasets(results_dir=results_dir)
        elapsed = time.time() - start
        all_results.append({'method': 'MDS', 'time': elapsed})
    else:
        print("\n⚠️  Skipping MDS (slow method)")

    # 4. Isomap
    print("\n" + "="*80)
    print("4/6: RUNNING ISOMAP")
    print("="*80)
    start = time.time()
    isomap.run_isomap_on_datasets(results_dir=results_dir)
    elapsed = time.time() - start
    all_results.append({'method': 'Isomap', 'time': elapsed})

    if not quick_mode:
        # 5. t-SNE
        print("\n" + "="*80)
        print("5/6: RUNNING t-SNE (SLOW)")
        print("="*80)
        start = time.time()
        tsne.run_tsne_on_datasets(results_dir=results_dir)
        elapsed = time.time() - start
        all_results.append({'method': 't-SNE', 'time': elapsed})

        # 6. UMAP
        if UMAP_AVAILABLE:
            print("\n" + "="*80)
            print("6/6: RUNNING UMAP")
            print("="*80)
            start = time.time()
            umap_dr.run_umap_on_datasets(results_dir=results_dir)
            elapsed = time.time() - start
            all_results.append({'method': 'UMAP', 'time': elapsed})
        else:
            print("\n⚠️  UMAP not available (install with: pip install umap-learn)")
    else:
        print("\n⚠️  Skipping t-SNE and UMAP (slow methods)")

    # Create comparisons
    print("\n" + "="*80)
    print("CREATING COMPARISON PLOTS")
    print("="*80)

    for dataset_name in datasets:
        create_comparison_plot(dataset_name, results_dir)

    # Summary
    create_summary_table(all_results, results_dir)

    print("\n" + "="*80)
    print("✅ ALL DONE!")
    print("="*80)
    print()
    print("Results saved in results/ directory:")
    print(f"  • {len(list(results_dir.glob('*.png')))} PNG visualizations")
    print(f"  • {len(list(results_dir.glob('*.csv')))} CSV files")
    print()
    print("Key files:")
    print("  • comparison_*.png - Side-by-side comparison of all methods")
    print("  • summary_all_methods.csv - Timing and metrics")
    print()
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dimensionality reduction comparison')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick mode: skip slow methods (MDS, t-SNE, UMAP)')
    parser.add_argument('--dataset', type=str, choices=['swiss_roll', 'faces', 'newsgroups'],
                       help='Run on specific dataset only')

    args = parser.parse_args()

    datasets_to_run = None
    if args.dataset:
        dataset_map = {'swiss_roll': 'Swiss Roll', 'faces': 'Faces', 'newsgroups': 'Newsgroups'}
        datasets_to_run = [dataset_map[args.dataset]]

    run_all_algorithms(datasets_to_run=datasets_to_run, quick_mode=args.quick)