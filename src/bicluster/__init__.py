import numpy as np
from scipy.stats import norm

class GaussianAsymmetricLBM:
    """
    Gaussian Latent Block Model (LBM) for a non-symmetric square matrix A,
    where row and column indices share the same K clusters (Z_i = W_i).

    Uses the EM algorithm with asymmetric block parameters (mu_rs != mu_sr).
    """

    def __init__(self, K, max_iter=100, tol=1e-4):
        """
        :param K: Number of clusters (groups) for both rows and columns.
        :param max_iter: Maximum number of EM iterations.
        :param tol: Tolerance for convergence (based on log-likelihood).
        """
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.mu = None
        self.sigma2 = None
        self.pi = None # Prior probability for Z_i (row/column group assignment)
        self.log_likelihood_history = []
        self.N = None

    def _initialize_parameters(self, A):
        """Initializes model parameters: means, variances, and priors."""
        self.N = A.shape[0]

        # 1. Initialize Block Parameters (mu, sigma2)
        # K x K matrices for mu_rs and sigma2_rs (non-symmetric in indices)
        # Randomly initialize mu (means) and sigma2 (variances)
        # Random initialization is often performed based on K-means initial partitioning,
        # but here we use a simple random start for simplicity.
        self.mu = np.random.uniform(np.min(A), np.max(A), size=(self.K, self.K))
        # Ensure variance is positive
        self.sigma2 = np.random.uniform(0.1, 1.0, size=(self.K, self.K))

        # 2. Initialize Prior Cluster Probabilities (pi)
        # Since Z_i = W_i, we only need one set of priors, pi_k = P(Z_i=k)
        self.pi = np.ones(self.K) / self.K

    def _calculate_complete_log_likelihood(self, A, tau_i, tau_ij):
        """
        Calculates the complete-data log-likelihood (used for convergence check).
        """
        L = 0.0
        for r in range(self.K):
            for s in range(self.K):
                # Likelihood contribution from block (r, s)
                # P(A_ij | Z_i=r, Z_j=s) * P(Z_i=r) * P(Z_j=s)
                
                # The term (tau_ij * log(P(A_ij|...))) is the expectation part
                # The Gaussian log-likelihood term
                log_prob_A_ij = norm.logpdf(A, loc=self.mu[r, s], scale=np.sqrt(self.sigma2[r, s]))
                
                # Sum of E[log L]
                L += np.sum(tau_ij[:, :, r, s] * log_prob_A_ij)
                
        # Add the expected log prior P(Z_i=k)
        for k in range(self.K):
             L += np.sum(tau_i[:, k] * np.log(self.pi[k]))

        return L

    def _e_step(self, A):
        """
        Expectation Step: Calculate posterior probabilities (soft assignments).
        
        tau_i[i, r] = P(Z_i = r | A, Theta)
        tau_ij[i, j, r, s] = P(Z_i = r, Z_j = s | A, Theta)
        """
        # tau_i is the soft membership for node i: N x K matrix (P(Z_i=k))
        # We need to calculate P(Z_i=k | A, Theta)
        
        # log_tau_i is a temporary matrix for log-probabilities
        log_tau_i = np.zeros((self.N, self.K))

        # P(A_ij | Z_i=r, Z_j=s) - N x N x K x K
        log_prob_A = np.zeros((self.N, self.N, self.K, self.K))
        
        # Pre-calculate the log-likelihood of each observation A_ij belonging to each block (r, s)
        for r in range(self.K):
            for s in range(self.K):
                log_prob_A[:, :, r, s] = norm.logpdf(A, loc=self.mu[r, s], scale=np.sqrt(self.sigma2[r, s]))

        # Calculate P(Z_i=k | A, Theta) which is proportional to P(A | Z_i=k, Theta) * P(Z_i=k)
        # P(A | Z_i=k) is proportional to:
        # P(Z_i=k) * [ Product_j P(A_ij | Z_i=k, Z_j) * Product_j P(A_ji | Z_j, Z_i=k) ]
        
        # Iteratively calculate the expected log-likelihood for each node i
        for i in range(self.N):
            for k in range(self.K):
                # E[ log P(A_i | Z_i=k) ]
                # log P(Z_i=k)
                log_p_i_given_k = np.log(self.pi[k])
                
                # Sum over all j of E[log P(A_ij | Z_i=k, Z_j) + log P(A_ji | Z_j, Z_i=k)]
                for j in range(self.N):
                    # Sum over all possible groups m for Z_j
                    log_p_ij_sum = 0.0
                    for m in range(self.K):
                        # P(Z_j=m | A) (approximated by pi_m or previous tau_i) * # [ P(A_ij | Z_i=k, Z_j=m) * P(A_ji | Z_j=m, Z_i=k) ]
                        
                        # Simplified approximation: we use the prior pi_m for Z_j, 
                        # which simplifies the structure but is standard in LBM implementation.
                        # The full expectation calculation requires iterative updates, 
                        # but we use the simpler factorization for tractability.
                        
                        log_prob_ij = log_prob_A[i, j, k, m] # P(A_ij | Z_i=k, Z_j=m)
                        log_prob_ji = log_prob_A[j, i, m, k] # P(A_ji | Z_j=m, Z_i=k)
                        
                        # Accumulate terms: P(Z_j=m) * likelihood(A_ij, A_ji)
                        # We use log-sum-exp to handle small probabilities
                        log_p_ij_sum = np.logaddexp(log_p_ij_sum, 
                                                    np.log(self.pi[m]) + log_prob_ij + log_prob_ji)

                    log_p_i_given_k += log_p_ij_sum
                
                log_tau_i[i, k] = log_p_i_given_k
        
        # Normalize to get posterior tau_i: P(Z_i=k | A, Theta)
        log_tau_i_max = np.max(log_tau_i, axis=1, keepdims=True)
        tau_i = np.exp(log_tau_i - log_tau_i_max)
        tau_i = tau_i / np.sum(tau_i, axis=1, keepdims=True)
        
        # Calculate tau_ij[i, j, r, s] = P(Z_i = r, Z_j = s | A, Theta) 
        # For simplicity and computational stability, we approximate this using:
        # P(Z_i=r, Z_j=s | A) ~ P(Z_i=r | A) * P(Z_j=s | A)
        tau_ij = np.einsum('ir, js -> ijsr', tau_i, tau_i)

        return tau_i, tau_ij

    def _m_step(self, A, tau_i, tau_ij):
        """
        Maximization Step: Update parameters (pi, mu, sigma2) using soft assignments.
        """
        # 1. Update Priors (pi_k)
        # pi_k = (1/N) * sum_i P(Z_i=k | A)
        self.pi = np.mean(tau_i, axis=0)

        # 2. Update Means (mu_rs) and Variances (sigma2_rs)
        for r in range(self.K):
            for s in range(self.K):
                # Numerator: sum_i sum_j P(Z_i=r, Z_j=s | A) * A_ij
                # Use einsum for vectorized weighted sum: sum_{ij} tau_ij * A
                numerator = np.sum(tau_ij[:, :, r, s] * A)

                # Denominator: sum_i sum_j P(Z_i=r, Z_j=s | A)
                denominator = np.sum(tau_ij[:, :, r, s])
                
                if denominator > 1e-8:
                    # Update mu_rs
                    self.mu[r, s] = numerator / denominator
                    
                    # Update sigma2_rs
                    # Numerator: sum_{ij} tau_ij * (A_ij - mu_rs)^2
                    weighted_sq_diff = tau_ij[:, :, r, s] * (A - self.mu[r, s])**2
                    self.sigma2[r, s] = np.sum(weighted_sq_diff) / denominator
                    
                    # Ensure variance is not zero
                    self.sigma2[r, s] = max(self.sigma2[r, s], 1e-6)
                else:
                    # If denominator is tiny, keep old values (or re-initialize)
                    pass


    def fit(self, A):
        """
        Runs the EM algorithm to fit the LBM to data A.
        :param A: The square, non-symmetric input matrix.
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError("Input matrix A must be square.")

        self._initialize_parameters(A)
        
        # Initialize tau_i for the first E-step
        tau_i = np.ones((self.N, self.K)) / self.K
        tau_ij = np.einsum('ir, js -> ijsr', tau_i, tau_i)

        for iteration in range(self.max_iter):
            # E-Step
            tau_i, tau_ij = self._e_step(A)
            
            # M-Step
            self._m_step(A, tau_i, tau_ij)
            
            # Calculate and check convergence
            current_log_likelihood = self._calculate_complete_log_likelihood(A, tau_i, tau_ij)
            self.log_likelihood_history.append(current_log_likelihood)
            
            if iteration > 0:
                delta = current_log_likelihood - self.log_likelihood_history[-2]
                if abs(delta) < self.tol * abs(current_log_likelihood):
                    print(f"EM converged after {iteration + 1} iterations.")
                    break
        
        self.tau_i = tau_i # Store final soft assignments
        return self


