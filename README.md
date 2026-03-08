# Medical Q&A Semantic Search System

A lightweight semantic search service over a curated medical and health Q&A corpus, featuring fuzzy clustering, a cluster-bucketed semantic cache, and a FastAPI service.

## Dataset

~2500 medical Q&A documents across 11 health categories:
- Symptoms, Treatment, Prevention, Medications
- Mental Health, Nutrition, Chronic Disease
- Emergency Medicine, Diagnosis, Infectious Disease, Genetics

No external downloads required — corpus is built-in.

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### Run in order:

```bash
python -m scripts.01_embed_and_index        # Build corpus + ChromaDB index
python -m scripts.02_fuzzy_cluster          # Fuzzy clustering (k=10)
python -m scripts.02b_save_cluster_model    # Save cluster model for inference
python -m scripts.03_explore_threshold      # (Optional) threshold analysis
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000/docs** to test interactively.

## API Endpoints

### POST /query
```json
{"query": "What are the symptoms of diabetes?", "top_k": 5}
```
**Cache miss response:**
```json
{
  "query": "What are the symptoms of diabetes?",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": null,
  "result": "Top 5 medical results for...",
  "dominant_cluster": 3
}
```
**Cache hit response** (after asking "How do I know if I have diabetes?"):
```json
{
  "query": "How do I know if I have diabetes?",
  "cache_hit": true,
  "matched_query": "What are the symptoms of diabetes?",
  "similarity_score": 0.9134,
  "result": "Top 5 medical results for...",
  "dominant_cluster": 3
}
```

### GET /cache/stats
```json
{
  "total_entries": 5,
  "hit_count": 3,
  "miss_count": 5,
  "hit_rate": 0.375,
  "threshold": 0.88
}
```

### DELETE /cache
Flushes all entries and resets stats.

## Docker

```bash
docker-compose up --build
```
