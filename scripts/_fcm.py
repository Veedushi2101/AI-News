"""
Shared Fuzzy C-Means implementation used by both
02_fuzzy_cluster.py and 02b_save_cluster_model.py.
"""
import numpy as np


class FuzzyCMeans:
    """Bezdek Fuzzy c-Means — see 02_fuzzy_cluster.py for design notes."""

    def __init__(self, n_clusters, m=2.0, max_iter=150, tol=1e-5, random_state=42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)

    def fit(self, X):
        n, d = X.shape
        k = self.n_clusters
        U = self.rng.dirichlet(np.ones(k), size=n)

        for iteration in range(self.max_iter):
            U_old = U.copy()
            Um = U ** self.m
            V = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-300)

            dists = np.zeros((n, k))
            for i in range(k):
                diff = X - V[i]
                dists[:, i] = np.sum(diff ** 2, axis=1)

            zero_mask = dists < 1e-10
            exp = 2.0 / (self.m - 1.0)
            new_U = np.zeros((n, k))

            for i in range(k):
                ratio = dists / (dists[:, i:i+1] + 1e-300)
                new_U[:, i] = 1.0 / ((ratio ** exp).sum(axis=1) + 1e-300)

            new_U[zero_mask.any(axis=1)] = 0.0
            for j in range(n):
                if zero_mask[j].any():
                    new_U[j, zero_mask[j]] = 1.0 / zero_mask[j].sum()

            U = new_U
            if np.linalg.norm(U - U_old) < self.tol:
                break

        self.U_ = U
        self.V_ = V
        self.labels_ = U.argmax(axis=1)
        return self

    def fuzzy_partition_coefficient(self):
        return float((self.U_ ** 2).sum() / self.U_.shape[0])
