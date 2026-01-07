"""
Comparison Framework for Dimensionality Reduction Methods

Compares LLE (from scratch) with course lab implementations
"""

import numpy as np
import pandas as pd
from time import time
from lle_implementation import LocallyLinearEmbedding
from course_methods import CourseLabMethods


class ManifoldLearningComparator:
    """
    Comprehensive comparison of dimensionality reduction methods

    Compares:
    - LLE (from-scratch implementation)
    - Course lab methods (PCA, MDS, Isomap, Kernel PCA, t-SNE, UMAP)
    """

    def __init__(self, n_components=2):
        """
        Parameters
        ----------
        n_components : int
            Number of dimensions in output space
        """
        self.n_components = n_components
        self.results = {}
        self.lab_methods = CourseLabMethods()

    def compare_all(self, X, params=None):
        """
        Run all methods and collect results

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        params : dict, optional
            Parameters for each method:
            - n_neighbors: for LLE, Isomap, UMAP
            - tsne_perplexity: for t-SNE
            - umap_n_neighbors: for UMAP

        Returns
        -------
        results : dict
            Dictionary with results for each method
        """
        if params is None:
            params = {
                'n_neighbors': 12,
                'tsne_perplexity': 30,
                'umap_n_neighbors': 15
            }

        N, D = X.shape
        print("="*80)
        print("DIMENSIONALITY REDUCTION COMPARISON")
        print("="*80)
        print(f"Dataset: {N} samples × {D} dimensions → {self.n_components} dimensions")
        print("="*80 + "\n")

        # 1. LLE (our from-scratch implementation)
        print("1. LLE (From Scratch Implementation)")
        print("-" * 80)
        start = time()
        lle = LocallyLinearEmbedding(n_components=self.n_components,
                                     n_neighbors=params['n_neighbors'])
        Y_lle = lle.fit_transform(X)
        t_lle = time() - start
        print(f"  Time: {t_lle:.4f}s\n")

        self.results['LLE'] = {
            'embedding': Y_lle,
            'time': t_lle,
            'lab': 'Our Implementation',
            'type': 'nonlinear_local'
        }

        # 2. PCA (Lab 2)
        print("2. PCA (Lab 2)")
        print("-" * 80)
        start = time()
        res_pca = self.lab_methods.lab2_pca(X, self.n_components)
        t_pca = time() - start
        res_pca['time'] = t_pca
        self.results['PCA'] = res_pca
        print(f"  Explained variance: {res_pca['explained_variance']:.4f}")
        print(f"  Time: {t_pca:.4f}s\n")

        # 3. MDS (Lab 2)
        print("3. Classical MDS (Lab 2)")
        print("-" * 80)
        start = time()
        res_mds = self.lab_methods.lab2_mds(X, self.n_components)
        t_mds = time() - start
        res_mds['time'] = t_mds
        self.results['MDS'] = res_mds
        print(f"  Stress: {res_mds['stress']:.2f}")
        print(f"  Time: {t_mds:.4f}s\n")

        # 4. Isomap (Lab 3)
        print("4. Isomap (Lab 3)")
        print("-" * 80)
        start = time()
        res_isomap = self.lab_methods.lab3_isomap(X, self.n_components,
                                                   params['n_neighbors'])
        t_isomap = time() - start
        res_isomap['time'] = t_isomap
        self.results['Isomap'] = res_isomap
        print(f"  Reconstruction error: {res_isomap['reconstruction_error']:.6f}")
        print(f"  Time: {t_isomap:.4f}s\n")

        # 5. Kernel PCA (Lab 4)
        print("5. Kernel PCA (Lab 4)")
        print("-" * 80)
        start = time()
        res_kpca = self.lab_methods.lab4_kernel_pca(X, self.n_components)
        t_kpca = time() - start
        res_kpca['time'] = t_kpca
        self.results['KernelPCA'] = res_kpca
        print(f"  Kernel: RBF")
        print(f"  Time: {t_kpca:.4f}s\n")

        # 6. t-SNE (Lab 5)
        print("6. t-SNE (Lab 5)")
        print("-" * 80)
        start = time()
        res_tsne = self.lab_methods.lab5_tsne(X, self.n_components,
                                              params['tsne_perplexity'])
        t_tsne = time() - start
        res_tsne['time'] = t_tsne
        self.results['t-SNE'] = res_tsne
        print(f"  KL divergence: {res_tsne['kl_divergence']:.4f}")
        print(f"  Time: {t_tsne:.4f}s\n")

        # 7. UMAP (Lab 5)
        print("7. UMAP (Lab 5)")
        print("-" * 80)
        start = time()
        res_umap = self.lab_methods.lab5_umap(X, self.n_components,
                                              params['umap_n_neighbors'])
        t_umap = time() - start
        res_umap['time'] = t_umap
        self.results['UMAP'] = res_umap
        print(f"  Time: {t_umap:.4f}s\n")

        print("="*80)
        print("✓ COMPARISON COMPLETE")
        print("="*80)

        return self.results

    def get_summary_table(self):
        """
        Create summary DataFrame

        Returns
        -------
        df : pandas DataFrame
            Summary of all methods with timing and metadata
        """
        data = {
            'Method': [],
            'Time (s)': [],
            'Source': [],
            'Type': []
        }

        for name, res in self.results.items():
            data['Method'].append(name)
            data['Time (s)'].append(res['time'])
            data['Source'].append(res['lab'])
            data['Type'].append(res['type'])

        df = pd.DataFrame(data)
        df = df.sort_values('Time (s)')
        df = df.reset_index(drop=True)
        return df

    def save_all_embeddings(self, prefix='embedding'):
        """
        Save all embeddings to CSV files

        Parameters
        ----------
        prefix : str
            Prefix for output filenames
        """
        for method_name, result in self.results.items():
            filename = f"{prefix}_{method_name.lower().replace('-', '').replace(' ', '_')}.csv"
            pd.DataFrame(result['embedding'], 
                        columns=['dim1', 'dim2']).to_csv(filename, index=False)
            print(f"Saved: {filename}")
