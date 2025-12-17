# `GaussianAsymmetricSBM`

## Gaussian Asymmetric Stochastic Block Model (SBM)

This repository contains a robust implementation of the
Expectation-Maximization (EM) algorithm for the **Gaussian Asymmetric
Stochastic Block Model (SBM)**. This model is designed to cluster the
nodes of a directed, weighted network (represented by an adjacency
matrix A) where the edge weights $A_{ij}$ are assumed to follow a
Gaussian distribution parameterized by the sender's block (r) and the
receiver's block (s).

The implementation features vectorized EM steps, numerically stable
log-sum-exp, calculation of the Evidence Lower Bound (ELBO) for
convergence tracking, and multiple intelligent initialization
strategies to avoid local optima.

## Installation and Dependencies

This code relies on standard scientific Python libraries. To install this package after cloning it with its dependencies:

```bash
uv pip install .

```

## Model Description

The model assumes that the network's $N$ nodes are partitioned into $K$ blocks, where the block assignment $Z_i$ for node $i$ is the same for its role as a row (sender) and column (receiver).

The observed edge weight $A_{ij}$ for nodes $i \neq j$ is modeled as:

where $Z_i, Z_j \in \{1, \dots, K\}$. The diagonal elements $A_{ii}$ (self-loops) are explicitly excluded from the likelihood calculation in both the E and M steps.

The model parameters $\Theta = \{\mu, \sigma^2, \pi\}$ are learned via the EM algorithm, which iteratively maximizes the Evidence Lower Bound (ELBO).

## How to Use

The primary class is `GaussianAsymmetricSBM`, and the primary function for running stability experiments is `estimate_number_of_hillclimbs`.

### 1. Generating Example Data (Setup)
First, create a sample directed network matrix A.

```python
# python setup_example.py

import numpy as np
from bicluster import GaussianAsymmetricSBM 

# Parameters for the synthetic network
N = 100  # Number of nodes
K_true = 3  # True number of clusters
block_size = N // K_true
K_indices = np.repeat(np.arange(K_true), block_size)

# Define block means (Asymmetric, Gaussian)
# Example: Cluster 0 talks to 1 a lot (mean 10), but 1 talks to 0 less (mean 3)
MU_TRUE = np.array([
    [15, 10, 5],
    [ 3, 12, 8],
    [ 6,  7, 14]
])
SIGMA_TRUE = 2.0 # Fixed variance for simplicity

A_synth = np.zeros((N, N))

# Generate data block by block
for r in range(K_true):
    for s in range(K_true):
        rows_r = np.where(K_indices == r)[0]
        cols_s = np.where(K_indices == s)[0]
        
        # Fill the block with Gaussian noise
        A_synth[np.ix_(rows_r, cols_s)] = np.random.normal(
            loc=MU_TRUE[r, s], 
            scale=SIGMA_TRUE, 
            size=(len(rows_r), len(cols_s))
        )

# Ensure diagonal is masked (not part of the model)
np.fill_diagonal(A_synth, 0)

```

### 2. Fitting the Model (Single Run)

To fit the model, instantiate the class and call `.fit()`.

```python
# Use the robust Spectral initialization (recommended)
K_MODEL = 3
sbm = GaussianAsymmetricSBM(K=K_MODEL, max_iter=100)
sbm.fit(A_synth, init='spectral')

# Access results
final_elbo = sbm.log_likelihood_history[-1]
soft_assignments = sbm.tau_i
mu_estimates = sbm.mu

print(f"Final ELBO: {final_elbo:.2f}")
print("Estimated Mean Matrix (mu):\n", np.round(mu_estimates, 2))

```

### 3. Assessing Stability and Confidence (Recommended)

To find the optimal solution and determine which node assignments are
stable, run the `estimate_number_of_hillclimbs` utility. This
repeatedly runs the EM algorithm from different initial conditions
(random seeds) and tracks the maximum log-likelihood (ELBO) achieved
and the variability of node assignments.

```python
from bicluster import estimate_number_of_hillclimbs

NUM_ATTEMPTS = 50
K_TEST = 3 # Hypothesized number of clusters

scores, predictions, entropy = estimate_number_of_hillclimbs(
    A=A_synth, 
    K=K_TEST, 
    num_attempts=NUM_ATTEMPTS, 
    init='spectral' # Use spectral initialization for the best results
)

# Find the best ELBO achieved across all attempts
best_score = np.max(scores)

# Nodes with low entropy have confident, stable assignments
low_entropy_nodes = np.where(entropy < 0.1)[0] 

print(f"\nMax ELBO achieved in {NUM_ATTEMPTS} attempts: {best_score:.2f}")
print(f"Number of confident node assignments (Entropy < 0.1): {len(low_entropy_nodes)}")
# 

```

##️ Initialization Methods (`init` parameter)

The choice of initialization is critical for EM algorithms, which are prone to converging to local optima. The `fit` method supports three modes:

| `init` Value | Method | Description | Recommendation |
| --- | --- | --- | --- |
| `'spectral'` | **Spectral Biclustering** | Uses `sklearn.cluster.SpectralBiclustering` on the matrix A to find a highly structured initial partition. | **Highly Recommended** (Best starting point). |
| `'kmeans'` | **K-Means Clustering** | Clusters nodes based on their concatenated in/out degree profiles (`[A_i, A^T_i]`). | Good (Very robust fallback). |
| `None` | **Random / Summary Stats** | Random initialization of \mu and \sigma^2, and uniform \tau_i. | **Not Recommended** (Highly susceptible to local optima). |

---

## FAQ

### 1. What does the "Hill Climb" utility achieve?
The **hill climb** utility (`estimate_number_of_hillclimbs`) runs the EM algorithm multiple times from different initializations. The EM algorithm is a "hill climbing" procedure on the likelihood surface. By running it many times, you achieve two goals:

* **Find the Global Optimum:** The run that yields the highest ELBO score is considered the best fit.
* **Assess Robustness:** The `entropy` output shows which node assignments are unstable, meaning they change depending on the starting point. High entropy suggests the node is an *outlier* or lies near a *boundary* between clusters.

### 2. Why does the model exclude diagonal elements?
The diagonal elements $A_{ii}$ represent self-loops or self-interaction values. The SBM is fundamentally a model of **relationships between distinct nodes**. Excluding $A_{ii}$ ensures that the clustering is driven purely by the block-level interactions $\mu_{rs}$ for $r \ne s$, which is the standard practice for modeling relational data.

### 3. What is the Evidence Lower Bound (ELBO)?
The ELBO, computed by `_calculate_log_likelihood`, is a measure of model fitness used in variational inference methods like the EM algorithm. It is composed of three parts:

$$\text{ELBO} = \mathbb{E}[\log P(A|Z)] + \mathbb{E}[\log P(Z|\pi)] - \mathbb{E}[\log Q(Z)]$$

It is guaranteed to increase at every step of the EM algorithm, making it the perfect metric to track convergence.

### 4. Why did I get a `ValueError` from `SpectralBiclustering`?
If you encounter an error like `ValueError: n_best=3 must be <= n_components=2.`, it means your `n_components` parameter is too low for the number of clusters K you requested. This implementation handles that by automatically setting `n_components` to be at least $\lceil \log_2(K) \rceil$, ensuring the spectral embedding has enough dimensions to distinguish all $K$ groups.