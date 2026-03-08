"""
Search engine for Medical Q&A semantic search.
"""
from __future__ import annotations

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma_db")
CLUSTER_META_PATH = os.path.join(DATA_DIR, "cluster_metadata.json")
MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "medical_qa"
N_CLUSTERS = 10
TOP_K = 5


class SearchEngine:
    def __init__(self):
        print("Loading embedding model …")
        self.model = SentenceTransformer(MODEL_NAME)

        print("Loading ChromaDB …")
        self.chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma.get_collection(COLLECTION_NAME)

        print("Loading cluster metadata …")
        with open(CLUSTER_META_PATH) as f:
            self.cluster_meta = json.load(f)

        centres_path = os.path.join(DATA_DIR, "cluster_centres.npy")
        umap_path = os.path.join(DATA_DIR, "umap_reducer.pkl")

        self._has_cluster_model = os.path.exists(centres_path)
        if self._has_cluster_model:
            import pickle
            self.cluster_centres = np.load(centres_path)
            with open(umap_path, "rb") as f:
                self.umap_reducer = pickle.load(f)
            self.FCM_M = 2.0
        else:
            print("  Warning: run 02b_save_cluster_model.py first for cluster membership.")
            self.cluster_centres = None

    def embed(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return vec.astype(np.float32)

    def get_membership(self, vec: np.ndarray) -> np.ndarray:
        if self.cluster_centres is not None:
            x_umap = self.umap_reducer.transform(vec.reshape(1, -1))[0]
            x_umap = x_umap / (np.linalg.norm(x_umap) + 1e-12)
            dists = np.array([np.sum((x_umap - c) ** 2) for c in self.cluster_centres])
            exp = 2.0 / (self.FCM_M - 1.0)
            membership = np.zeros(N_CLUSTERS)
            for i in range(N_CLUSTERS):
                if dists[i] < 1e-10:
                    membership[i] = 1.0
                    return membership
                ratio = dists / (dists[i] + 1e-300)
                membership[i] = 1.0 / (ratio ** exp).sum()
            return membership
        else:
            return np.ones(N_CLUSTERS) / N_CLUSTERS

    def search(self, query: str, top_k: int = TOP_K):
        query_vec = self.embed(query)
        membership = self.get_membership(query_vec)
        dominant_cluster = int(np.argmax(membership))

        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        lines = [f"Top {len(docs)} medical results for: \"{query}\"\n"]
        for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), 1):
            similarity = 1 - dist
            snippet = doc[:250].replace("\n", " ")
            lines.append(
                f"[{rank}] similarity={similarity:.3f} | "
                f"category={meta.get('label_name', 'unknown')} | "
                f"{snippet}…"
            )

        return "\n".join(lines), query_vec, membership, dominant_cluster


_engine: SearchEngine | None = None


def get_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine()
    return _engine
