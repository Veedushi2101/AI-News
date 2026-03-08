"""
02b_save_cluster_model.py
=========================
Saves the UMAP reducer and FCM cluster centres so the FastAPI service
can assign cluster membership to new medical queries at runtime.
"""

import os
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import normalize
import umap
from scripts._fcm import FuzzyCMeans

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EMBEDDINGS_PATH = os.path.join(DATA_DIR, "embeddings.npy")
UMAP_DIM = 30
N_CLUSTERS = 10
FCM_M = 2.0
RANDOM_STATE = 42

print("Loading embeddings …")
embeddings = np.load(EMBEDDINGS_PATH)

print("Fitting UMAP reducer …")
reducer = umap.UMAP(
    n_components=UMAP_DIM, n_neighbors=15, min_dist=0.0,
    metric="cosine", random_state=RANDOM_STATE
)
X_reduced = reducer.fit_transform(embeddings)
X_reduced = normalize(X_reduced, norm="l2")

print("Fitting FCM …")
fcm = FuzzyCMeans(n_clusters=N_CLUSTERS, m=FCM_M,
                  max_iter=150, tol=1e-5, random_state=RANDOM_STATE)
fcm.fit(X_reduced)

umap_path = os.path.join(DATA_DIR, "umap_reducer.pkl")
centres_path = os.path.join(DATA_DIR, "cluster_centres.npy")

with open(umap_path, "wb") as f:
    pickle.dump(reducer, f)
np.save(centres_path, fcm.V_)

print(f"✓ Saved UMAP reducer → {umap_path}")
print(f"✓ Saved cluster centres ({fcm.V_.shape}) → {centres_path}")
print("Done.")
