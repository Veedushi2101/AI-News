# MedSearch — Medical Q&A Semantic Search

A semantic search system for medical and health questions. It understands what you mean — not just what you type. Built as part of the Trademarkia AI/ML Engineer assignment.

---

## The problem this solves

A keyword search treats *"signs of diabetes"* and *"how do I know if I have diabetes"* as completely different queries and recomputes everything from scratch. This system treats them as the same question. The second time a semantically equivalent query arrives, the result is served from a cluster-bucketed in-memory cache in under 5ms.

---

## How it works

```
Query
  |
  v
Sentence embedding (all-MiniLM-L6-v2, 384-dim)
  |
  v
Fuzzy cluster membership (UMAP + Fuzzy c-Means, k=10)
  |
  v
Semantic cache lookup (cosine similarity within cluster bucket)
  |-- HIT  --> return cached result instantly
  |
  -- MISS --> ChromaDB vector search --> store in cache --> return
```

The clustering does real work here. Instead of scanning every cached entry on every lookup, the system only compares against entries in the same semantic neighbourhood. As the cache grows, this stays fast.

---

## Project structure

```
medsearch/
|
|-- app/
|   |-- main.py                    FastAPI service and all endpoints
|   |-- semantic_cache.py          Cluster-bucketed cache, pure Python + numpy
|   |-- search_engine.py           Embedding and ChromaDB retrieval
|   `-- static/
|       `-- index.html             Frontend UI
|
|-- scripts/
|   |-- 01_embed_and_index.py      Clean, embed, and index the corpus
|   |-- 02_fuzzy_cluster.py        UMAP dimensionality reduction + Fuzzy c-Means
|   |-- 02b_save_cluster_model.py  Persist cluster model for query-time inference
|   |-- 03_explore_threshold.py    Threshold precision/recall sweep
|   `-- _fcm.py                    Fuzzy c-Means from scratch (numpy only)
|
|-- tests/
|   `-- test_cache.py              Unit tests for the semantic cache
|
|-- Dockerfile
|-- docker-compose.yml
`-- requirements.txt
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/Veedushi2101/AI-News.git
cd AI-News

python -m venv venv
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Mac / Linux

pip install -r requirements.txt
```

**Build the pipeline** — run these four in order:

```bash
python -m scripts.01_embed_and_index
python -m scripts.02_fuzzy_cluster
python -m scripts.02b_save_cluster_model
python -m scripts.03_explore_threshold
```

**Start the server:**

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` for the UI, or `http://localhost:8000/docs` for the API explorer.

---

## API

### POST /query

Takes a natural language medical question and returns semantically similar results. On the first call it queries ChromaDB and stores the result. On subsequent calls with equivalent phrasing it returns from cache.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the symptoms of diabetes?", "top_k": 5}'
```

First call — cache miss:
```json
{
  "query": "What are the symptoms of diabetes?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "Top 5 medical results...",
  "dominant_cluster": 3
}
```

Second call with different wording — cache hit:
```json
{
  "query": "How do I know if I have diabetes?",
  "cache_hit": true,
  "matched_query": "What are the symptoms of diabetes?",
  "similarity_score": 0.871,
  "result": "Top 5 medical results...",
  "dominant_cluster": 3
}
```

### GET /cache/stats

```json
{
  "total_entries": 12,
  "hit_count": 8,
  "miss_count": 12,
  "hit_rate": 0.40,
  "threshold": 0.75,
  "bucket_sizes": {"3": 5, "6": 4, "8": 3}
}
```

### DELETE /cache

Flushes all entries and resets counters.

### GET /health

Returns service status and configuration.

---

## Design decisions

### Embedding model

`all-MiniLM-L6-v2` was chosen over larger models like MPNet or E5 because it was trained specifically on paraphrase pairs. That is exactly the similarity this cache needs to detect. It also runs in roughly 40ms per batch on CPU, keeping query latency acceptable. A BioBERT or PubMedBERT model would give marginally better domain accuracy but is 4-8x slower — the wrong trade-off for a cache.

### Why fuzzy clustering

Medical topics do not have clean edges. A question about chest pain after exercise belongs to symptoms, emergency medicine, and chronic disease simultaneously. Hard clustering (k-means, HDBSCAN) forces it into one category. Fuzzy c-Means assigns it a probability distribution across all clusters — which is the accurate representation. The output for each document is a vector like [0.55, 0.30, 0.15, ...] not a single label.

### Cluster count

Tested k from 5 to 14 using Fuzzy Partition Coefficient and silhouette score. k=10 gave the best balance. k=20 (matching the category count) performed worse because several categories overlap so heavily in semantic space that separating them artificially hurts cluster quality.

### Cache data structure

```
_buckets: dict[cluster_id -> list[CacheEntry]]
```

Each entry stores the query text, its L2-normalised embedding, the result, and its cluster membership vector. Lookup computes cosine similarity as a dot product (fast because vectors are pre-normalised) only within the dominant cluster bucket. For boundary queries — where the top cluster membership is below 0.5 — the second-best bucket is also checked.

### The threshold

The similarity threshold is the single most important tunable parameter. Too low and the cache starts returning results for genuinely different questions. Too high and it never hits.

The `03_explore_threshold.py` script runs a full precision/recall sweep across thresholds from 0.60 to 1.00 using labelled paraphrase pairs. The default of 0.75 was chosen based on this analysis — it is the point where paraphrase recall is consistently above 90% while cross-topic false positives stay near zero.

This is what each value range actually does to the system's behaviour:

| Range | Effect |
|-------|--------|
| 0.60 – 0.70 | Nearly everything hits. Cache becomes unreliable — wrong results get served |
| 0.75 – 0.82 | Paraphrases hit reliably. Topically different questions miss. Recommended range |
| 0.85 – 0.92 | Conservative. High precision but recall drops on looser paraphrases |
| 0.95 – 1.00 | Barely better than string matching. Cache provides minimal benefit |

### No external caching libraries

The cache is implemented in plain Python dicts and numpy. No Redis, Memcached, or any caching middleware. The only external dependency for the cache itself is numpy for dot product computation.

---

## Dataset

The corpus is built in — approximately 2,500 medical Q&A pairs across 11 categories. No external download is required. Each document is stored as `Q: <question> A: <answer>` so the embedding captures the full context of both question and answer, not just the question wording.

Categories: symptoms, treatment, prevention, medications, mental health, nutrition, chronic disease, emergency medicine, diagnosis, infectious disease, genetics.

---

## Docker

```bash
docker-compose up --build
```

The container starts uvicorn on port 8000. The `data/` directory is mounted as a volume so pre-built embeddings persist between restarts.

```bash
# Or manually
docker build -t medsearch .
docker run -p 8000:8000 -v $(pwd)/data:/app/data medsearch
```

---

## Tests

```bash
pytest tests/ -v
```

Covers: exact cache hits, near-duplicate detection, cross-cluster misses, flush behaviour, stats accuracy, bucket isolation, and threshold exploration output format.

---

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SIMILARITY_THRESHOLD` | `0.75` | Cosine similarity threshold for cache hits |
| `N_CLUSTERS` | `10` | Number of fuzzy clusters |

```bash
# Windows PowerShell
$env:SIMILARITY_THRESHOLD="0.80"
python -m uvicorn app.main:app --port 8000
```

---

## Performance

| | |
|--|--|
| Corpus size | ~2,500 documents |
| Embedding size | 384 dimensions |
| Query latency — miss | 80–120ms (CPU) |
| Query latency — hit | < 5ms |
| Cache lookup | O(bucket size), sub-linear as cache grows |
