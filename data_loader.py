"""
Data Loader for LLE Project

Loads three datasets:
1. Swiss Roll - Synthetic manifold
2. ORL/Yale Faces - Image data (classic LLE benchmark)
3. 20 Newsgroups - Text data

Usage:
    from data_loader import load_swiss_roll, load_faces, load_newsgroups

    X_swiss, y_swiss, info_swiss = load_swiss_roll()
    X_faces, y_faces, info_faces = load_faces()
    X_news, y_news, info_news = load_newsgroups()
"""

import numpy as np
from sklearn.datasets import make_swiss_roll, fetch_olivetti_faces, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def load_swiss_roll(n_samples=2000, noise=0.1, random_state=42):
    """
    Load Swiss Roll synthetic manifold dataset

    A 2D manifold embedded in 3D space - classic manifold learning benchmark.
    Used in original LLE paper (Roweis & Saul, 2000).

    Parameters
    ----------
    n_samples : int, default=2000
        Number of samples to generate
    noise : float, default=0.1
        Standard deviation of Gaussian noise
    random_state : int, default=42
        Random seed for reproducibility

    Returns
    -------
    X : array, shape (n_samples, 3)
        3D coordinates of Swiss Roll points (standardized)
    y : array, shape (n_samples,)
        Color values (position along manifold) - for visualization only
    info : dict
        Dataset metadata
    """
    print("Loading Swiss Roll dataset...")

    # Generate Swiss Roll
    X, color = make_swiss_roll(n_samples=n_samples, noise=noise, 
                                random_state=random_state)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    info = {
        'name': 'Swiss Roll',
        'n_samples': n_samples,
        'n_features': 3,
        'intrinsic_dim': 2,
        'noise': noise,
        'description': '2D manifold rolled in 3D space',
        'source': 'sklearn.datasets.make_swiss_roll'
    }

    print(f"✓ Swiss Roll loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]}D")
    return X_scaled, color, info


def load_faces(n_samples=400, random_state=42):
    """
    Load Olivetti Faces dataset

    Face images - similar to ORL/Yale faces used in LLE paper.
    Each face is 64×64 pixels = 4096 dimensions.
    40 subjects with 10 images each.

    Parameters
    ----------
    n_samples : int, default=400
        Number of face images to use (max 400)
    random_state : int, default=42
        Random seed

    Returns
    -------
    X : array, shape (n_samples, 4096)
        Flattened face images (standardized)
    y : array, shape (n_samples,)
        Subject labels (0-39)
    info : dict
        Dataset metadata
    """
    print("Loading Olivetti Faces dataset...")

    # Fetch faces
    faces_data = fetch_olivetti_faces(shuffle=True, random_state=random_state)

    # Select subset if requested
    X = faces_data.data[:n_samples]
    y = faces_data.target[:n_samples]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    info = {
        'name': 'Olivetti Faces',
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_subjects': len(np.unique(y)),
        'image_shape': (64, 64),
        'description': 'Face images - 40 subjects, 10 images each',
        'source': 'sklearn.datasets.fetch_olivetti_faces',
        'similar_to': 'ORL/Yale faces used in LLE paper'
    }

    print(f"✓ Faces loaded: {X_scaled.shape[0]} images, {X_scaled.shape[1]}D (64×64 pixels)")
    return X_scaled, y, info


def load_newsgroups(n_samples=2000, categories=None, max_features=1000, random_state=42):
    """
    Load 20 Newsgroups text dataset

    Text documents from 20 different newsgroups.
    Converted to TF-IDF features for dimensionality reduction.

    Parameters
    ----------
    n_samples : int, default=2000
        Approximate number of documents to load
    categories : list, optional
        Subset of categories to load. If None, uses 5 diverse categories
    max_features : int, default=1000
        Maximum number of TF-IDF features
    random_state : int, default=42
        Random seed

    Returns
    -------
    X : array, shape (n_samples, max_features)
        TF-IDF feature vectors (standardized)
    y : array, shape (n_samples,)
        Category labels
    info : dict
        Dataset metadata including category names
    """
    print("Loading 20 Newsgroups dataset...")

    # Use subset of categories for faster loading and clearer separation
    if categories is None:
        categories = [
            'comp.graphics',
            'sci.space', 
            'rec.sport.baseball',
            'talk.politics.mideast',
            'alt.atheism'
        ]

    # Fetch newsgroups
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=random_state
    )

    # Convert to TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(newsgroups.data).toarray()
    y = newsgroups.target

    # Shuffle and limit samples
    np.random.seed(random_state)
    n_available = X.shape[0]
    if n_samples < n_available:
        indices = np.random.choice(n_available, n_samples, replace=False)
        X = X[indices]
        y = y[indices]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    info = {
        'name': '20 Newsgroups',
        'n_samples': X_scaled.shape[0],
        'n_features': X_scaled.shape[1],
        'n_categories': len(categories),
        'categories': categories,
        'description': 'Text documents from newsgroups (TF-IDF features)',
        'source': 'sklearn.datasets.fetch_20newsgroups',
        'max_features': max_features
    }

    print(f"✓ Newsgroups loaded: {X_scaled.shape[0]} documents, {X_scaled.shape[1]}D TF-IDF features")
    print(f"  Categories: {categories}")
    return X_scaled, y, info


# Convenience function to load all datasets at once
def load_all_datasets():
    """
    Load all three datasets at once

    Returns
    -------
    datasets : dict
        Dictionary with keys 'swiss_roll', 'faces', 'newsgroups'
        Each value is a tuple (X, y, info)
    """
    print("="*80)
    print("LOADING ALL DATASETS")
    print("="*80)
    print()

    datasets = {}

    # Load each dataset
    datasets['swiss_roll'] = load_swiss_roll()
    print()

    datasets['faces'] = load_faces()
    print()

    datasets['newsgroups'] = load_newsgroups()
    print()

    print("="*80)
    print("✓ ALL DATASETS LOADED")
    print("="*80)

    return datasets


if __name__ == '__main__':
    """Test data loading"""

    # Test individual loading
    print("Testing individual dataset loading...")
    print()

    X_swiss, y_swiss, info_swiss = load_swiss_roll()
    print(f"Swiss Roll shape: {X_swiss.shape}")
    print()

    X_faces, y_faces, info_faces = load_faces()
    print(f"Faces shape: {X_faces.shape}")
    print()

    X_news, y_news, info_news = load_newsgroups()
    print(f"Newsgroups shape: {X_news.shape}")
    print()

    # Test loading all at once
    print("\n" + "="*80)
    print("Testing load_all_datasets()...")
    print("="*80)
    datasets = load_all_datasets()

    print("\nDataset summary:")
    for name, (X, y, info) in datasets.items():
        print(f"  {info['name']:20s}: {X.shape[0]:4d} samples × {X.shape[1]:5d} features")
