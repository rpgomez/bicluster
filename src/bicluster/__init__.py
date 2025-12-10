import numpy as np
from scipy.stats import norm
from sklearn.cluster import KMeans

class GaussianAsymmetricSBM:
    """
    Gaussian Stochastic Block Model (SBM) for a non-symmetric matrix A,
    where item clusters (Z_i) are the same for rows/columns (R=S=K).
    
    Model: A_ij ~ N(mu_rs, sigma_rs^2) for i != j.
    Uses the EM algorithm with soft assignments (tau_ik).
    """

    def __init__(self, K, max_iter=100, tol=1e-5):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol
        self.N = None
        
        # Parameters to be learned
        self.mu = None     # K x K block means (mu_rs)
        self.sigma2 = None # K x K block variances (sigma2_rs)
        self.pi = None     # K vector of cluster priors (pi_k)
        self.tau_i = None  # N x K soft assignments (tau_ik)
        self.log_likelihood_history = []

    def _initialize_parameters_robust(self, A):
        """Initializes parameters using K-Means for a stable start."""
        self.N = A.shape[0]

        # 1. Create feature matrix by concatenating row and column profiles
        # Feats_i = [A_i, A^T_i]
        Features = np.hstack((A, A.T))
        
        # 2. Run K-Means to get initial hard assignments
        # Use a consistent random state for reproducibility
        kmeans = KMeans(n_clusters=self.K, random_state=42, n_init='auto')
        initial_hard_assignments = kmeans.fit_predict(Features)
        
        # 3. Use hard assignments to initialize soft assignments (tau_i)
        self.tau_i = np.zeros((self.N, self.K))
        for i in range(self.N):
            # Start with 90% confidence in the K-Means cluster, 
            # and spread 10% uniformly across others (smooths initialization)
            k_assign = initial_hard_assignments[i]
            self.tau_i[i, :] = 0.1 / (self.K - 1)
            self.tau_i[i, k_assign] = 0.9

        # 4. Initialize Priors (pi_k) based on K-Means counts
        self.pi = np.mean(self.tau_i, axis=0)
        
        # 5. Initialize Block Parameters (mu, sigma2) based on K-Means blocks
        self.mu = np.zeros((self.K, self.K))
        self.sigma2 = np.zeros((self.K, self.K))
        diag_mask = ~np.eye(self.N, dtype=bool)

        for r in range(self.K):
            for s in range(self.K):
                # Identify items belonging to group r and s
                rows_r = np.where(initial_hard_assignments == r)[0]
                cols_s = np.where(initial_hard_assignments == s)[0]
                
                # Get the submatrix A[r, s]
                block_A = A[np.ix_(rows_r, cols_s)]
                
                # Exclude diagonal if r == s
                if r == s:
                    # Flatten the block, excluding the diagonal terms
                    indices = np.where(diag_mask[np.ix_(rows_r, cols_s)])
                    block_data = block_A[indices]
                else:
                    block_data = block_A.flatten()

                # Calculate parameters from the initial partition
                if len(block_data) > 1:
                    self.mu[r, s] = np.mean(block_data)
                    self.sigma2[r, s] = np.var(block_data)
                else:
                    # Fallback for empty/tiny blocks
                    self.mu[r, s] = np.mean(A)
                    self.sigma2[r, s] = 1.0 # Use a safe, large variance

        # Enforce variance floor (Crucial Fix)
        self.sigma2 = np.maximum(self.sigma2, 0.5) # Using 0.5 as a safe minimum

    def _initialize_parameters(self, A):
        """Initializes parameters and soft assignments (tau_i)."""
        self.N = A.shape[0]
        
        # 1. Initialize Block Parameters (mu, sigma2)
        # Random start for means and variances
        self.mu = np.random.uniform(np.min(A), np.max(A), size=(self.K, self.K))
        self.sigma2 = np.random.uniform(0.1, 1.0, size=(self.K, self.K))

        # 2. Initialize Prior Cluster Probabilities (pi)
        self.pi = np.ones(self.K) / self.K
        
        # 3. Initialize Soft Assignments (tau_i)
        # Using uniform initial soft assignments
        self.tau_i = np.ones((self.N, self.K)) / self.K

    def log_sum_exp_numpy(self,vals,axis=None):
        """
        Computes log(sum(exp(vals))) in a numerically stable way.

        Args:
            vals (np.ndarray or list): A sequence of log values.

        Returns:
            float: The log of the sum of the exponentials.
        """

        if axis is None:
            c = np.max(vals)

            # 2. Compute the log-sum-exp using the shift
            # The term (vals - c) ensures the largest exponent is 0.
            result = c + np.log(np.sum(np.exp(vals - c)))
        else:
            c = np.max(vals,axis=axis,keepdims=True)
            c_true = np.max(vals,axis=axis)

            result = c_true + np.log(np.sum(np.exp(vals - c),axis=axis))
        return result

    def _e_step(self, A):
        """
        Expectation Step: Computes soft assignments tau_ik = P(Z_i=k | A, Theta).
        """
        log_tau_i = np.zeros((self.N, self.K))
        
        # Pre-calculate log-likelihood P(A_ij | Z_i=r, Z_j=s) for all (i,j) and (r,s)
        # log_prob_A[i, j, r, s]

        N = self.N
        K = self.K

        idx = np.arange(N)
        
        loc = self.mu.reshape(1,1,K,K)
        scale = np.sqrt(self.sigma2).reshape(1,1,K,K)
        log_prob_A = norm.logpdf(A.reshape(N,N,1,1),loc=loc,scale=scale)
        
        # 1. Start with the log prior: log P(Z_i=k)
        log_p_i_given_k = np.zeros((N,K))
        log_p_i_given_k[:] = np.log(self.pi) # shape N x K


        # 2. Contribution from i as a SOURCE (A_ij)
        # Sum over all other nodes j != i and their possible groups m
        terms = log_prob_A[:,:,:,:] + np.log(self.tau_i).reshape(1,N,1,K) # shape N x N x K x K

        # zero out the ith entry since we're not making use of A[i,i]
        terms[idx,idx] = -np.inf
        log_contrib_source = self.log_sum_exp_numpy(terms,axis=3) # down to shape N x N x K
        log_contrib_source[idx,idx] = 0 # log_contrib_source[i] == -inf which hoses me.
        log_p_i_given_k += log_contrib_source.sum(axis=1) # down to shape N x K

        # 3. Contribution from i as a SINK (A_ji)
        # Sum over all other nodes j != i and their possible groups m

        terms = log_prob_A[:,:,:,:] + np.log(self.tau_i).reshape(N,1,K,1) # shape N x N x K x K

        # zero out the ith entry since we're not making use of A[i,i]
        terms[idx,idx] = -np.inf
        log_contrib_sink = self.log_sum_exp_numpy(terms,axis=2) # down to shape N x N x K
        log_contrib_sink[idx,idx] = 0 # log_contrib_source[i] == -inf which hoses me.
        log_p_i_given_k += log_contrib_sink.sum(axis=0) # down to shape N x K

        log_tau_i = log_p_i_given_k

        # Normalize log probabilities to get tau_i (posterior P(Z_i=k | A))
        log_tau_i_max = np.max(log_tau_i, axis=1, keepdims=True)
        self.tau_i = np.exp(log_tau_i - log_tau_i_max)
        self.tau_i = self.tau_i / np.sum(self.tau_i, axis=1, keepdims=True)
        
        # Avoid division by zero in case of an item having zero probability everywhere
        self.tau_i[np.isnan(self.tau_i)] = 1.0 / self.K


    def _m_step(self, A):
        """
        Maximization Step: Updates parameters (pi, mu, sigma2) using tau_i.
        
        Crucially excludes diagonal terms (i=j) in all summations.
        """

        N = self.N
        K = self.K
        
        # Create a mask to exclude the diagonal terms (i=j)
        diag_mask = ~np.eye(self.N, dtype=bool) # shape N x N
        
        # 1. Update Priors (pi_k)
        self.pi = np.mean(self.tau_i, axis=0)

        # 2. Update Means (mu_rs) and Variances (sigma2_rs)
        tau_ij_rs = self.tau_i.reshape(N,1,K,1)*self.tau_i.reshape(1,N,1,K) # shape N x N x K x K
        weighted_tau = diag_mask.reshape(N,N,1,1)*tau_ij_rs # shape N x N x K x K
        denominator = weighted_tau.sum(axis=(0,1)) # shape K x K
        numerator_mu = tau_ij_rs*A.reshape(N,N,1,1) # shape N x N x K x K
        numerator_mu = numerator_mu.sum(axis=(0,1)) # shape K x K
        ratio = numerator_mu/denominator

        # update only when we have enough evidence (denominator > 1e-8)
        self.mu[denominator > 1e-8] = ratio[denominator>1e-8]
        numerator2 = tau_ij_rs*(A**2).reshape(N,N,1,1) # shape N x N x K x K
        numerator2 = numerator2.sum(axis=(0,1))
        ratio2 = numerator2/denominator
        self.sigma2[denominator > 1e-8] = (ratio2 - self.mu**2)[denominator> 1e-8]
        self.sigma2[self.sigma2 < 1e-3] = 1e-3

    def _calculate_log_likelihood(self, A):
        """
        Computes the Evidence Lower Bound (ELBO).
        ELBO = E[log P(A|Z)] + E[log P(Z)] + H(Z)
        """
        N = self.N
        K = self.K
        eps = 1e-10  # Small epsilon to prevent log(0)

        # 1. Expand parameters for broadcasting
        # Shape: (1, 1, K, K)
        loc = self.mu.reshape(1, 1, K, K)
        scale = np.sqrt(self.sigma2).reshape(1, 1, K, K)

        # 2. Compute Log Probability of every edge A_ij belonging to every block combination (r, s)
        # Shape: (N, N, K, K)
        # Note: This matches the memory footprint of your E-step
        log_prob_A = norm.logpdf(A.reshape(N, N, 1, 1), loc=loc, scale=scale)

        # 3. Compute the Joint Posterior Weight for every pair (i, j)
        # Weight[i, j, r, s] = tau[i, r] * tau[j, s]
        # Shape: (N, N, K, K)
        weight_ij_rs = self.tau_i.reshape(N, 1, K, 1) * self.tau_i.reshape(1, N, 1, K)

        # 4. Zero out the diagonal (self-loops are ignored in the model)
        # We assume the model does not generate A_ii
        idx = np.arange(N)
        weight_ij_rs[idx, idx, :, :] = 0.0

        # --- TERM 1: Expected Log Likelihood of A ---
        # Sum_{i!=j} Sum_{r,s} P(Z_i=r, Z_j=s) * log P(A_ij | theta_rs)
        term_data = np.sum(weight_ij_rs * log_prob_A)

        # --- TERM 2: Expected Log Prior ---
        # Sum_i Sum_k P(Z_i=k) * log(pi_k)
        term_prior = np.sum(self.tau_i * np.log(self.pi + eps))

        # --- TERM 3: Entropy (Model Confidence) ---
        # - Sum_i Sum_k P(Z_i=k) * log P(Z_i=k)
        term_entropy = -np.sum(self.tau_i * np.log(self.tau_i + eps))

        return term_data + term_prior + term_entropy

    def fit(self, A):
        # ... (initialization code) ...
        self._initialize_parameters_robust(A)
        self.log_likelihood_history = []
        prev_ll = -np.inf

        for iteration in range(self.max_iter):
            tau_old = self.tau_i.copy()

            # E-Step
            self._e_step(A)
            
            # M-Step
            self._m_step(A)
            
            # --- NEW: Calculate and Store Log Likelihood ---
            ll = self._calculate_log_likelihood(A)
            self.log_likelihood_history.append(ll)
            
            # Check for convergence based on likelihood (optional but recommended)
            if abs(ll - prev_ll) < self.tol:
                print(f"Converged at iter {iteration} with LL: {ll:.4f}")
                break
            prev_ll = ll
        
        return self

