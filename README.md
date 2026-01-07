# Locally Linear Embedding (LLE)
## Deep Dive into the Algorithm, Mathematics, and Implementation

> **Based on:** Roweis, S. T., & Saul, L. K. (2000). *Nonlinear dimensionality reduction by locally linear embedding*. Science, 290(5500), 2323-2326.

---

## 📋 Table of Contents

1. [Introduction](#introduction)
2. [The Problem](#the-problem)
3. [The Key Intuition](#the-key-intuition)
4. [Mathematical Framework](#mathematical-framework)
5. [The LLE Algorithm](#the-lle-algorithm)
6. [Properties and Guarantees](#properties-and-guarantees)
7. [Implementation Details](#implementation-details)
8. [Comparison with Other Methods](#comparison-with-other-methods)
9. [Results and Analysis](#results-and-analysis)
10. [References](#references)

---

## Introduction

**Locally Linear Embedding (LLE)** is an unsupervised learning algorithm that computes low-dimensional, neighborhood-preserving embeddings of high-dimensional inputs. Unlike previous methods, **LLE recovers global nonlinear structure from locally linear fits**.

### Key Characteristics

- **Nonlinear** - Can unwrap curved manifolds
- **Unsupervised** - No labels required
- **Global optimum** - No local minima in optimization
- **Sparse matrices** - Computationally efficient
- **Single parameter** - Only K (number of neighbors) needs tuning

---

## 🔍 The Problem

### Dimensionality Reduction

Given:
- **N** data points \( \{\mathbf{X}_i\}_{i=1}^N \)
- Each point \( \mathbf{X}_i \in \mathbb{R}^D \) (high-dimensional)
- Data lies on or near a **smooth manifold** of intrinsic dimension \( d \ll D \)

Goal:
- Find low-dimensional representations \( \{\mathbf{Y}_i\}_{i=1}^N \)
- Each \( \mathbf{Y}_i \in \mathbb{R}^d \) (low-dimensional)
- Preserve the **local geometry** of the manifold

### The Challenge

Traditional linear methods (PCA, classical MDS) fail on nonlinear manifolds:

```
3D Swiss Roll (nonlinear manifold)
      ╱│
     ╱ │     PCA projects here →  ●●●●●● (mess)
    ╱  │                          ●●●●●●
   ╱   │     LLE unfolds here →  ┌──────┐ (clean)
  ╱    │                          │ ●●●● │
 ╱     │                          └──────┘
```

**Why PCA fails:** It preserves distances along straight lines, not along the curved manifold.

---

## 💡 The Key Intuition

### Locally Linear, Globally Nonlinear

**Core Insight:** Although a manifold may be globally nonlinear, **each small neighborhood is approximately linear**.

#### Analogy: The Earth

- 🌍 **Globally:** Earth is curved (sphere)
- 📍 **Locally:** Your neighborhood looks flat

You can use a flat map for your city, even though Earth is round!

### LLE's Strategy

1. **Assume:** Each data point can be **reconstructed** as a linear combination of its neighbors
2. **Compute:** Optimal reconstruction weights for each point
3. **Key property:** These weights are **invariant** to rotations, translations, and rescalings
4. **Exploit:** Use the same weights to find low-dimensional coordinates

> **"Think globally, fit locally"** - The weights capture intrinsic geometry that transfers from high-D to low-D

---

## 📐 Mathematical Framework

### Notation

| Symbol | Meaning | Dimension |
|--------|---------|-----------|
| \( N \) | Number of data points | scalar |
| \( D \) | Ambient (input) dimensionality | scalar |
| \( d \) | Intrinsic (output) dimensionality | scalar (\( d \ll D \)) |
| \( K \) | Number of neighbors | scalar |
| \( \mathbf{X}_i \) | i-th data point (input) | \( \mathbb{R}^D \) |
| \( \mathbf{Y}_i \) | i-th embedding (output) | \( \mathbb{R}^d \) |
| \( W_{ij} \) | Reconstruction weight | scalar |
| \( \mathbf{W} \) | Weight matrix | \( \mathbb{R}^{N \times N} \) |

### Assumptions

1. **Manifold assumption:** Data lies on or near a smooth \( d \)-dimensional manifold
2. **Sampling assumption:** Manifold is well-sampled (enough neighbors)
3. **Local linearity:** Each neighborhood lies on/near a locally linear patch

---

## 🔢 The LLE Algorithm

LLE consists of **three steps**, each with a clear geometric interpretation.

---

### **Step 1: Find Neighbors**

For each data point \( \mathbf{X}_i \), identify its \( K \) nearest neighbors.

#### Methods:
- **Euclidean distance:** \( d(\mathbf{X}_i, \mathbf{X}_j) = \|\mathbf{X}_i - \mathbf{X}_j\|_2 \)
- **Fixed radius:** All points within distance \( r \)
- **Normalized dot product:** For sparse/text data

#### Neighbor Matrix:
Define \( \eta(i) \) = set of neighbors of \( \mathbf{X}_i \)

---

### **Step 2: Compute Reconstruction Weights**

#### Goal:
Reconstruct each point from its neighbors:

\[
\mathbf{X}_i \approx \sum_{j \in \eta(i)} W_{ij} \mathbf{X}_j
\]

#### Cost Function:
Minimize the **reconstruction error**:

\[
\varepsilon(\mathbf{W}) = \sum_{i=1}^N \left\| \mathbf{X}_i - \sum_{j=1}^N W_{ij} \mathbf{X}_j \right\|^2
\]

#### Constraints:
1. **Locality:** \( W_{ij} = 0 \) if \( j \notin \eta(i) \) (only neighbors contribute)
2. **Sum-to-one:** \( \sum_{j=1}^N W_{ij} = 1 \) for each \( i \) (affine invariance)

---

#### **Detailed Solution (Closed Form)**

For each data point \( \mathbf{X}_i \) with neighbors \( \{\mathbf{X}_j\}_{j \in \eta(i)} \):

##### **Step 2a: Center neighbors**
\[
\mathbf{Z}_j = \mathbf{X}_j - \mathbf{X}_i \quad \text{for } j \in \eta(i)
\]

##### **Step 2b: Compute local covariance (Gram matrix)**
\[
C_{jk} = \mathbf{Z}_j^T \mathbf{Z}_k = (\mathbf{X}_j - \mathbf{X}_i)^T (\mathbf{X}_k - \mathbf{X}_i)
\]

This is a \( K \times K \) matrix measuring correlations between neighbors.

##### **Step 2c: Solve constrained least squares**

The weights are given by:

\[
W_{ij} = \frac{\sum_{k \in \eta(i)} C_{jk}^{-1}}{\sum_{l,m \in \eta(i)} C_{lm}^{-1}}
\]

**In practice:**
1. Compute \( \mathbf{C}^{-1} \) (invert the \( K \times K \) matrix)
2. Sum all entries: \( s = \sum_{jk} C_{jk}^{-1} \)
3. Sum each row: \( r_j = \sum_k C_{jk}^{-1} \)
4. Normalize: \( W_{ij} = r_j / s \)

##### **Regularization (if C is singular)**

If \( \mathbf{C} \) is nearly singular (collinear neighbors):

\[
\mathbf{C}_{\text{reg}} = \mathbf{C} + \text{reg} \cdot \text{tr}(\mathbf{C}) \cdot \mathbf{I}
\]

This adds a small ridge term proportional to the trace.

---

#### **Why This Works: Geometric Invariance**

The reconstruction weights have a crucial property:

**Theorem (Invariance):** For any data point and its neighbors, the optimal weights \( W_{ij} \) are **invariant** to:
- **Translations:** \( \mathbf{X}_i \to \mathbf{X}_i + \mathbf{t} \)
- **Rotations:** \( \mathbf{X}_i \to \mathbf{R}\mathbf{X}_i \)
- **Uniform scaling:** \( \mathbf{X}_i \to s\mathbf{X}_i \)

**Why this matters:** The weights capture **intrinsic geometry**, not the coordinate system. They characterize the local manifold structure and will work in any embedding!

**Translation invariance** is enforced by the sum-to-one constraint \( \sum_j W_{ij} = 1 \).

---

### **Step 3: Compute Embedding**

#### Goal:
Find low-dimensional coordinates \( \{\mathbf{Y}_i\}_{i=1}^N \) that best preserve the reconstruction weights.

#### Cost Function:
Minimize the **embedding error**:

\[
\Phi(\mathbf{Y}) = \sum_{i=1}^N \left\| \mathbf{Y}_i - \sum_{j=1}^N W_{ij} \mathbf{Y}_j \right\|^2
\]

**Note:** The weights \( W_{ij} \) are **fixed** from Step 2. We optimize over \( \mathbf{Y}_i \).

---

#### **Detailed Solution (Eigenvalue Problem)**

##### **Step 3a: Define the cost as a quadratic form**

The cost can be rewritten as:

\[
\Phi(\mathbf{Y}) = \sum_{i,j} M_{ij} (\mathbf{Y}_i \cdot \mathbf{Y}_j)
\]

where \( \mathbf{M} \) is the **LLE matrix**:

\[
M_{ij} = \delta_{ij} - W_{ij} - W_{ji} + \sum_{k=1}^N W_{ki} W_{kj}
\]

Or more compactly:

\[
\mathbf{M} = (\mathbf{I} - \mathbf{W})^T (\mathbf{I} - \mathbf{W})
\]

**Properties of \( \mathbf{M} \):**
- Symmetric: \( \mathbf{M} = \mathbf{M}^T \)
- Positive semi-definite: \( \mathbf{M} \succeq 0 \)
- Sparse: Only non-zero where neighbors exist
- Rank: \( N - 1 \) (one zero eigenvalue)

##### **Step 3b: Apply constraints**

To make the problem well-posed, we constrain:

1. **Centered:** \( \sum_{i=1}^N \mathbf{Y}_i = \mathbf{0} \) (remove translation freedom)
2. **Unit covariance:** \( \frac{1}{N} \sum_{i=1}^N \mathbf{Y}_i \mathbf{Y}_i^T = \mathbf{I}_d \) (remove scaling freedom)

##### **Step 3c: Solve eigenvalue problem**

\[
\mathbf{M} \mathbf{v} = \lambda \mathbf{v}
\]

**Solution:**
1. Find the \( d+1 \) **smallest eigenvalues** of \( \mathbf{M} \)
2. The smallest is \( \lambda_0 = 0 \) with eigenvector \( \mathbf{v}_0 = (1, 1, \ldots, 1)^T/\sqrt{N} \) → **discard this**
3. The next \( d \) eigenvectors \( \{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_d\} \) give the embedding:

\[
\mathbf{Y}_i = \begin{bmatrix} v_1(i) \\ v_2(i) \\ \vdots \\ v_d(i) \end{bmatrix}
\]

where \( v_k(i) \) is the \( i \)-th component of eigenvector \( \mathbf{v}_k \).

---

#### **Why Eigenvalues?**

The embedding cost can be written as:

\[
\Phi(\mathbf{Y}) = \sum_{k=1}^d \lambda_k
\]

where \( \lambda_k \) are eigenvalues of \( \mathbf{M} \). Minimizing \( \Phi \) means choosing coordinates corresponding to the **smallest eigenvalues** (after discarding the zero mode).

---

### **Algorithm Summary**

```
Input: Data {X_i}, number of neighbors K, target dimension d
Output: Embeddings {Y_i}

1. FOR each point i:
     Find K nearest neighbors → η(i)

2. FOR each point i:
     Compute local Gram matrix C_jk = (X_j - X_i)^T (X_k - X_i)
     Solve for weights: W_ij = Σ_k C_jk^(-1) / Σ_jk C_jk^(-1)
     Enforce: W_ij = 0 if j ∉ η(i)

3. Construct M = (I - W)^T (I - W)
   Find bottom d+1 eigenvectors of M
   Discard eigenvector #0
   Y_i = [v_1(i), v_2(i), ..., v_d(i)]^T

Return: {Y_i}
```

---

## ✨ Properties and Guarantees

### Optimization Guarantees

1. **Global optimum:** Both Step 2 and Step 3 find **global minima**
2. **No local minima:** Unlike neural networks or EM algorithms
3. **Single pass:** Algorithm runs in one pass (no iterations)
4. **Convex problems:** Both steps solve convex optimization problems

### Computational Complexity

| Step | Operation | Complexity |
|------|-----------|------------|
| Step 1 | Find K-NN for N points | \( O(DN^2) \) or \( O(DN \log N) \) with tree |
| Step 2 | Solve N systems of size K×K | \( O(NK^3) \) |
| Step 3 | Sparse eigenvalue problem | \( O(dN^2) \) or better with sparse solvers |

**Overall:** \( O(DN^2) \) dominated by neighbor search.

### Scalability

- **Sparse matrices:** \( \mathbf{M} \) has only \( O(NK) \) non-zero entries
- **Incremental:** Can add dimensions without recomputation
- **Parallel:** Step 2 can be parallelized (each point independent)

### Parameter Selection

**Only one parameter:** \( K \) (number of neighbors)

**Guidelines:**
- Too small \( K \): Captures noise, disconnected graph
- Too large \( K \): Loses local structure, expensive
- **Rule of thumb:** \( K \approx \log N \) to \( 2\log N \)
- **Constraint:** \( K > d \) (need more neighbors than target dimensions)

---

## 💻 Implementation Details

### Step-by-Step Implementation

#### **Step 1: Neighbor Finding**

```python
from sklearn.neighbors import NearestNeighbors

# Find K nearest neighbors
nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree')
nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)
indices = indices[:, 1:]  # Remove self
```

#### **Step 2: Weight Computation**

```python
import numpy as np

W = np.zeros((N, N))

for i in range(N):
    # Get neighbors
    neighbors = indices[i]
    X_neighbors = X[neighbors]

    # Center the neighbors
    Z = X_neighbors - X[i]  # Shape: (K, D)

    # Compute Gram matrix C = Z @ Z.T
    C = Z @ Z.T  # Shape: (K, K)

    # Regularization (if needed)
    trace = np.trace(C)
    C += reg * trace * np.eye(K)

    # Solve for weights
    # We need w such that C @ w = 1 and sum(w) = 1
    w = np.linalg.solve(C, np.ones(K))
    w /= w.sum()  # Normalize

    # Store weights
    W[i, neighbors] = w
```

#### **Step 3: Embedding via Eigendecomposition**

```python
from scipy.sparse.linalg import eigsh

# Construct M = (I - W)^T (I - W)
I = np.eye(N)
M = (I - W).T @ (I - W)

# Find bottom d+1 eigenvectors
# (We use d+2 to be safe, then discard the smallest)
eigenvalues, eigenvectors = eigsh(M, k=d+1, which='SM')

# Discard the first eigenvector (corresponding to λ=0)
Y = eigenvectors[:, 1:d+1]  # Shape: (N, d)
```

### Robust Mode (for Sparse Data)

For sparse/text data (like newsgroups), use higher regularization:

```python
# Standard: reg = 1e-3
# Robust: reg = 10.0

C_reg = C + reg * np.trace(C) * np.eye(K)
```

---

## 📊 Comparison with Other Methods

### Methods Compared in Original Paper

| Method | Type | Preserves | Optimization | Speed |
|--------|------|-----------|--------------|-------|
| **LLE** | Nonlinear | Local linear structure | Global (eigen) | Fast |
| **PCA** | Linear | Variance | Global (SVD) | Fastest |
| **MDS** | Linear | Pairwise distances | Global (eigen) | Slow |
| **Isomap** | Nonlinear | Geodesic distances | Global (eigen) | Medium |

### Modern Methods (Post-2000)

| Method | Year | Type | Preserves | Best For |
|--------|------|------|-----------|----------|
| **t-SNE** | 2008 | Nonlinear | Probability distributions | Visualization |
| **UMAP** | 2018 | Nonlinear | Topological structure | Large datasets |

---

### LLE vs PCA

**PCA (Linear):**
\[
\mathbf{Y} = \mathbf{U}^T \mathbf{X}
\]
- Finds principal directions of variance
- Linear projection → **fails on Swiss Roll**

**LLE (Nonlinear):**
\[
\mathbf{Y}_i \approx \sum_j W_{ij} \mathbf{Y}_j
\]
- Preserves local neighborhoods
- Nonlinear embedding → **unwraps Swiss Roll**

**Example:**

```
Swiss Roll (3D):              PCA (2D):                LLE (2D):
     ╱│                          ●●●                   ┌──────┐
    ╱ │                          ●●●  ← FAIL           │ ●●●● │ ← SUCCESS
   ╱  │                          ●●●                   └──────┘
  ╱   │
```

**Quote from paper:**
> "Projections of the data by principal component analysis (PCA) or classical MDS map faraway data points to nearby points in the plane, failing to identify the underlying structure of the manifold."

---

### LLE vs Isomap

**Main comparison in the paper!**

| Aspect | LLE | Isomap |
|--------|-----|--------|
| **Preserves** | Local linear structure | Geodesic distances |
| **Approach** | Reconstruction weights | Shortest paths on graph |
| **Computation** | Sparse matrix eigensolve | All-pairs shortest paths |
| **Complexity** | \( O(dN^2) \) | \( O(N^2 \log N + dN^2) \) |
| **Local/Global** | Local emphasis | Global emphasis |
| **Handles holes** | Better | Struggles if graph disconnected |

**Quote from paper:**
> "Isomap's embeddings are optimized to preserve geodesic distances between general pairs of data points, which can only be estimated by computing shortest paths through large sublattices of data. LLE takes a different approach, analyzing local symmetries, linear coefficients, and reconstruction errors instead of global constraints, pairwise distances, and stress functions."

**Both methods:**
- Nonlinear
- Global optimum
- Unwrap Swiss Roll
- Share the principle: "overlapping local neighborhoods → global geometry"

---

## 📈 Results and Analysis

### Datasets Used

#### 1. Swiss Roll (N=2000, D=3, d=2)
- **Purpose:** Test manifold unwrapping
- **Parameters:** K=20
- **Result:** ✅ **Perfect unwrapping**
- **PCA comparison:** ❌ PCA fails completely

```
Original 3D:           LLE 2D:              PCA 2D:
   ╱│                 ┌────────┐              ●●●
  ╱ │                 │  ●●●●  │              ●●● ← Mixed
 ╱  │      →          │  ●●●●  │      vs      ●●●   together
╱   │                 └────────┘              
                      Successfully             Failed to
                      unwrapped!               unwrap
```

#### 2. Faces (N=2000, D=560, d=2)
- **Purpose:** Real image data (20×28 grayscale)
- **Parameters:** K=12
- **Result:** ✅ **Meaningful embedding**
- **Observation:** Coordinates correspond to pose and expression

```
LLE Embedding Space:
        ↑ (smiling)
        │
Left ←──┼──→ Right
        │
        ↓ (neutral)
```

#### 3. Words (N=5000, D=31000, d=2)
- **Purpose:** Text data (word-document counts)
- **Parameters:** K=20, normalized dot product
- **Result:** ✅ **Semantic clustering**
- **Observation:** Similar words cluster together

---

### Quantitative Evaluation (from Paper)

The paper uses **residual variance**:

\[
\text{Residual Variance} = 1 - R^2(\hat{\mathbf{D}}_M, \mathbf{D}_Y)
\]

where:
- \( \mathbf{D}_Y \) = Euclidean distances in embedding
- \( \hat{\mathbf{D}}_M \) = Estimated manifold distances
- \( R \) = correlation coefficient

**Lower is better** (embedding preserves distances).

---

### Typical Results

| Dataset | K | Reconstruction Error | Quality |
|---------|---|---------------------|---------|
| Swiss Roll | 20 | ~0.04 | Excellent unwrapping |
| Faces | 12 | Medium | Meaningful coordinates |
| Newsgroups | 10 | High (sparse) | Requires robust mode |

---

### Common Issues and Solutions

#### Issue 1: Singular Gram Matrix

**Symptom:** Matrix \( \mathbf{C} \) is not invertible

**Cause:** 
- Neighbors are collinear
- K > D (more neighbors than dimensions)
- Numerical precision

**Solution:** Add regularization
```python
C += reg * np.trace(C) * np.eye(K)  # reg = 1e-3 typical
```

#### Issue 2: Disconnected Graph

**Symptom:** Some points have no path to others

**Cause:** K too small, or data has multiple components

**Solution:**
- Increase K
- Check connected components
- Apply LLE separately to each component

#### Issue 3: Sparse/Text Data

**Symptom:** High reconstruction error, poor embedding

**Cause:** TF-IDF vectors are sparse, high-dimensional

**Solution:** Use robust mode
```python
# Higher regularization
reg = 10.0  # instead of 1e-3

# Smaller K
K = 10  # instead of 20
```

---

## 🎯 Key Takeaways

### The Big Ideas

1. **Locally linear, globally nonlinear**
   - Manifolds are locally flat, even if globally curved
   - Capture local geometry with linear weights

2. **Weights are intrinsic**
   - Invariant to rotations, translations, scalings
   - Transfer from high-D to low-D

3. **Global optimum via local fits**
   - No local minima
   - "Think globally, fit locally"

4. **Eigenvalue problem**
   - Bottom eigenvectors = best embedding
   - Sparse, efficient computation

### When to Use LLE

✅ **Good for:**
- Nonlinear manifolds
- Well-sampled data
- Local structure important
- Need global optimum guarantee

❌ **Not ideal for:**
- Very sparse data (use t-SNE/UMAP)
- Disconnected manifolds
- Need to preserve global distances (use Isomap)

---

## 🚀 Running the Code

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run LLE only
python lle.py

# Run all comparisons
python main.py --quick

# Full comparison (slow)
python main.py
```

### Project Structure

```
project/
├── lle.py              # LLE implementation
├── pca.py              # PCA comparison
├── isomap.py           # Isomap comparison
├── tsne.py             # t-SNE comparison
├── umap_dr.py          # UMAP comparison
├── main.py             # Run all
└── results/            # Generated outputs
    ├── lle_*.png       # LLE visualizations
    ├── comparison_*.png # Side-by-side comparisons
    └── *.csv           # Embeddings
```

---

## 📚 References

### Original Paper

**Roweis, S. T., & Saul, L. K. (2000).** *Nonlinear dimensionality reduction by locally linear embedding.* Science, 290(5500), 2323-2326.

### Related Work

- **Tenenbaum, J. B., et al. (2000).** Isomap: A global geometric framework for nonlinear dimensionality reduction. Science, 290(5500), 2319-2323.

- **van der Maaten, L., & Hinton, G. (2008).** Visualizing data using t-SNE. Journal of machine learning research, 9(11).

- **McInnes, L., Healy, J., & Melville, J. (2018).** UMAP: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426.

### Course Materials

**Unsupervised Learning 2025**
- Manifold Learning
- Dimensionality Reduction
- Clustering and Density Estimation

---
