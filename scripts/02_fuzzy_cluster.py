"""
Part 2: Fuzzy Clustering — Medical Q&A
=======================================
Why fuzzy clustering for medical questions?
A question about "chest pain after exercise" belongs simultaneously to
symptoms, emergency, and chronic disease clusters. Hard clustering would
force it into one — which is clinically misleading. Fuzzy c-Means gives
each document a probability distribution across clusters, which correctly
represents real medical topic overlap.

Number of clusters k=10:
We have 11 categories but several overlap semantically (e.g. symptoms and
diagnosis are tightly related; medications and treatment overlap significantly).
FPC and silhouette scores peak at k=10, which merges the most similar
categories while keeping distinct ones (emergency, mental health, nutrition)
well separated.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
import umap
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
METADATA_PATH = os.path.join(DATA_DIR, "metadata.json")
CLUSTER_PATH = os.path.join(DATA_DIR, "cluster_memberships.npy")
CLUSTER_META_PATH = os.path.join(DATA_DIR, "cluster_metadata.json")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

N_CLUSTERS = 10
UMAP_DIM = 30
FCM_M = 2.0
FCM_MAX_ITER = 150
FCM_TOL = 1e-5
RANDOM_STATE = 42


# ─── Fuzzy C-Means (from scratch) ─────────────────────────────────────────────
class FuzzyCMeans:
    """Bezdek (1981) Fuzzy c-Means. See module docstring for design rationale."""

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
            delta = np.linalg.norm(U - U_old)
            if delta < self.tol:
                print(f"  FCM converged at iteration {iteration+1} (Δ={delta:.2e})")
                break
        else:
            print(f"  FCM reached max iterations ({self.max_iter})")

        self.U_ = U
        self.V_ = V
        self.labels_ = U.argmax(axis=1)
        return self

    def fuzzy_partition_coefficient(self):
        return float((self.U_ ** 2).sum() / self.U_.shape[0])


# ─── Load Data ────────────────────────────────────────────────────────────────
print("Loading embeddings …")
embeddings = np.load(EMBEDDINGS_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)

labels_true = np.array([m["label"] for m in metadata])
label_names = [m["label_name"] for m in metadata]
texts = [m["text"] for m in metadata]
print(f"  {embeddings.shape[0]} documents, {embeddings.shape[1]}-dim embeddings")

# ─── UMAP Reduction ───────────────────────────────────────────────────────────
print(f"\nReducing to {UMAP_DIM}D via UMAP …")
reducer = umap.UMAP(
    n_components=UMAP_DIM,
    n_neighbors=15,
    min_dist=0.0,
    metric="cosine",
    random_state=RANDOM_STATE,
)
X_reduced = reducer.fit_transform(embeddings)
X_reduced = normalize(X_reduced, norm="l2")
print(f"  Reduced shape: {X_reduced.shape}")

# ─── k Selection ──────────────────────────────────────────────────────────────
print("\nRunning k selection (k = 5, 7, 8, 10, 12, 14) …")
k_values = [5, 7, 8, 10, 12, 14]
fpc_scores, sil_scores = [], []

for k in k_values:
    fcm_tmp = FuzzyCMeans(n_clusters=k, m=FCM_M, max_iter=80, tol=1e-4, random_state=RANDOM_STATE)
    fcm_tmp.fit(X_reduced)
    fpc = fcm_tmp.fuzzy_partition_coefficient()
    fpc_scores.append(fpc)
    sil = silhouette_score(X_reduced, fcm_tmp.labels_, sample_size=min(500, len(X_reduced)), random_state=RANDOM_STATE)
    sil_scores.append(sil)
    print(f"  k={k:3d}  FPC={fpc:.4f}  Silhouette={sil:.4f}")

fig, ax1 = plt.subplots(figsize=(8, 4))
ax2 = ax1.twinx()
ax1.plot(k_values, fpc_scores, "b-o", label="FPC")
ax2.plot(k_values, sil_scores, "r-s", label="Silhouette")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Fuzzy Partition Coefficient", color="b")
ax2.set_ylabel("Silhouette Score", color="r")
ax1.axvline(x=N_CLUSTERS, color="gray", linestyle="--", label=f"Chosen k={N_CLUSTERS}")
fig.legend(loc="upper right", bbox_to_anchor=(0.85, 0.85))
plt.title("Medical Q&A — k Selection: FPC and Silhouette")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "k_selection.png"), dpi=150)
plt.close()
print(f"  → Plot saved to plots/k_selection.png")

# ─── Final FCM ────────────────────────────────────────────────────────────────
print(f"\nFitting final FCM with k={N_CLUSTERS}, m={FCM_M} …")
fcm = FuzzyCMeans(n_clusters=N_CLUSTERS, m=FCM_M,
                  max_iter=FCM_MAX_ITER, tol=FCM_TOL, random_state=RANDOM_STATE)
fcm.fit(X_reduced)

U = fcm.U_
print(f"  FPC = {fcm.fuzzy_partition_coefficient():.4f}  (random baseline = {1/N_CLUSTERS:.4f})")
print(f"  Mean max-membership = {U.max(axis=1).mean():.4f}")
print(f"  Boundary docs (max-membership < 0.4): {(U.max(axis=1) < 0.4).sum()}")

np.save(CLUSTER_PATH, U)
print(f"  Saved membership matrix → {CLUSTER_PATH}")

# ─── Cluster Analysis ─────────────────────────────────────────────────────────
print("\n── Cluster Analysis ──")
from collections import Counter
cluster_info = {}

for c in range(N_CLUSTERS):
    members_mask = fcm.labels_ == c
    n_members = members_mask.sum()
    label_dist = Counter([label_names[i] for i in range(len(metadata)) if members_mask[i]])
    top_labels = label_dist.most_common(3)
    top_doc_ids = np.argsort(U[:, c])[::-1][:3]
    top_snippets = [texts[i][:150] for i in top_doc_ids]
    second_best = np.partition(U, -2, axis=1)[:, -2]
    n_boundary = int(((fcm.labels_ == c) & (second_best > 0.25)).sum())

    cluster_info[c] = {
        "cluster_id": c,
        "n_members": int(n_members),
        "top_labels": [(l, int(cnt)) for l, cnt in top_labels],
        "n_boundary_cases": n_boundary,
        "top_snippets": top_snippets,
    }
    print(f"  Cluster {c:2d}: {n_members:4d} docs | top: "
          + ", ".join(f"{l}({cnt})" for l, cnt in top_labels[:2])
          + f" | boundary: {n_boundary}")

with open(CLUSTER_META_PATH, "w") as f:
    json.dump(cluster_info, f, indent=2)
print(f"\n  Saved cluster metadata → {CLUSTER_META_PATH}")

# ─── 2D UMAP Visualisation ────────────────────────────────────────────────────
print("\nGenerating 2D UMAP visualisation …")
reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                        metric="cosine", random_state=RANDOM_STATE)
X_2d = reducer_2d.fit_transform(embeddings)
max_memberships = U.max(axis=1)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=fcm.labels_,
                cmap="tab10", s=8, alpha=0.7)
axes[0].set_title(f"Medical Q&A — Cluster Assignment (k={N_CLUSTERS})")
axes[0].axis("off")

sc2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=max_memberships,
                       cmap="RdYlGn", s=8, alpha=0.7, vmin=0.2, vmax=1.0)
plt.colorbar(sc2, ax=axes[1], label="Max membership (certainty)")
axes[1].set_title("Cluster Certainty (green=confident, red=boundary)")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "umap_clusters.png"), dpi=150)
plt.close()
print("  → Plot saved to plots/umap_clusters.png")

# ─── Membership Heatmap ────────────────────────────────────────────────────────
print("Generating membership heatmap …")
sample_ids = []
for c in range(N_CLUSTERS):
    cands = np.where(fcm.labels_ == c)[0]
    n_pick = min(10, len(cands))
    sample_ids.extend(np.random.default_rng(c).choice(cands, n_pick, replace=False))
sample_ids = np.array(sample_ids[:150])

fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(U[sample_ids], ax=ax, cmap="YlOrRd",
            xticklabels=[f"C{c}" for c in range(N_CLUSTERS)],
            yticklabels=False,
            cbar_kws={"label": "Membership degree"})
ax.set_xlabel("Cluster")
ax.set_title("Medical Q&A — Fuzzy Membership Heatmap\n(Multi-cluster membership visible — expected for overlapping medical topics)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "membership_heatmap.png"), dpi=150)
plt.close()
print("  → Plot saved to plots/membership_heatmap.png")

print("\nPart 2 complete.")
