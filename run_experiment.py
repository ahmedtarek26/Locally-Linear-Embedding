"""
Main Experiment Script - Swiss Roll Comparison

Runs LLE and all course lab methods on Swiss Roll dataset
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import StandardScaler
from comparison_framework import ManifoldLearningComparator


def main():
    """Run complete experiment on Swiss Roll dataset"""

    print("="*80)
    print("LLE PROJECT - SWISS ROLL EXPERIMENT")
    print("="*80)
    print()

    # Generate Swiss Roll dataset
    print("📊 DATASET: SWISS ROLL MANIFOLD")
    print("-" * 80)

    n_samples = 2000
    noise = 0.1
    random_state = 42

    X, color = make_swiss_roll(n_samples=n_samples, noise=noise, 
                                random_state=random_state)

    print(f"Generated Swiss Roll:")
    print(f"  Samples: {n_samples}")
    print(f"  Dimensions: 3D (embedded in 3D space)")
    print(f"  Intrinsic dimension: 2D (rolled 2D surface)")
    print(f"  Noise: {noise}")
    print(f"  Random seed: {random_state}")
    print()
    print("What is Swiss Roll?")
    print("  - 2D surface rolled into 3D space")
    print("  - Points close on surface may be far in 3D")
    print("  - Tests ability to 'unroll' nonlinear structure")
    print("  - Used in original LLE paper (Roweis & Saul, 2000)")
    print()

    # Standardize data
    print("Preprocessing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Data standardized (zero mean, unit variance)")
    print()

    # Save dataset
    np.save('data/swiss_roll_data.npy', X)
    np.save('data/swiss_roll_color.npy', color)
    np.save('data/swiss_roll_scaled.npy', X_scaled)
    print("✓ Saved dataset to data/ directory")
    print()

    # Run comparison
    print("="*80)
    print("🔬 RUNNING COMPARISON")
    print("="*80)
    print()

    comparator = ManifoldLearningComparator(n_components=2)
    results = comparator.compare_all(
        X_scaled,
        params={
            'n_neighbors': 12,      # Same as LLE paper for faces
            'tsne_perplexity': 30,  # Standard t-SNE
            'umap_n_neighbors': 15  # Standard UMAP
        }
    )

    # Display summary
    print()
    print("="*80)
    print("📊 RESULTS SUMMARY")
    print("="*80)
    print()

    summary = comparator.get_summary_table()
    print(summary.to_string(index=False))
    print()

    # Save results
    print("="*80)
    print("💾 SAVING RESULTS")
    print("="*80)
    print()

    comparator.save_all_embeddings(prefix='results/embedding')
    summary.to_csv('results/comparison_summary.csv', index=False)
    print("Saved: results/comparison_summary.csv")
    print()

    print("="*80)
    print("✅ EXPERIMENT COMPLETE!")
    print("="*80)
    print()
    print("Files created:")
    print("  data/swiss_roll_*.npy - Dataset files")
    print("  results/embedding_*.csv - Embedding coordinates")
    print("  results/comparison_summary.csv - Timing summary")


if __name__ == '__main__':
    main()
