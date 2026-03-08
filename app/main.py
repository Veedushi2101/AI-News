"""
Part 4: FastAPI Service — Medical Q&A Semantic Search
======================================================
Start with:
  python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.semantic_cache import SemanticCache
from app.search_engine import get_engine, SearchEngine

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.88"))
N_CLUSTERS = int(os.getenv("N_CLUSTERS", "10"))

_cache: SemanticCache = None
_engine: SearchEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _cache, _engine
    print("Initialising Medical Search Engine …")
    _engine = get_engine()
    print("Initialising Semantic Cache …")
    _cache = SemanticCache(
        similarity_threshold=SIMILARITY_THRESHOLD,
        n_clusters=N_CLUSTERS,
        boundary_check=True,
    )
    print(f"✓ Medical Q&A Service ready (threshold={SIMILARITY_THRESHOLD})")
    yield


app = FastAPI(
    title="Medical Q&A Semantic Search API",
    description=(
        "Semantic search over medical and health questions with fuzzy clustering "
        "and a cluster-bucketed semantic cache. Covers symptoms, treatment, "
        "prevention, medications, mental health, nutrition, and more."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(
        ..., min_length=3, max_length=500,
        example="What are the symptoms of diabetes?"
    )
    top_k: int = Field(5, ge=1, le=10,
                       description="Number of results to retrieve on cache miss")


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: str
    dominant_cluster: int


class CacheStats(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float
    threshold: float
    bucket_sizes: dict


class FlushResponse(BaseModel):
    message: str


@app.post("/query", response_model=QueryResponse,
          summary="Search medical Q&A with semantic cache")
async def query(request: QueryRequest) -> QueryResponse:
    """
    Embed the medical query, check the semantic cache, and return results.

    - **Cache hit**: same or rephrased question seen before — returns instantly.
    - **Cache miss**: retrieves similar Q&A from the medical corpus and caches it.
    """
    if _engine is None or _cache is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    result_str, query_vec, membership, dominant_cluster = _engine.search(
        request.query, top_k=request.top_k
    )

    hit = _cache.lookup(query_vec, membership)

    if hit is not None:
        cached_entry, score = hit
        return QueryResponse(
            query=request.query,
            cache_hit=True,
            matched_query=cached_entry.query_text,
            similarity_score=round(score, 4),
            result=cached_entry.result,
            dominant_cluster=cached_entry.dominant_cluster,
        )

    _cache.store(
        query_text=request.query,
        query_vec=query_vec,
        result=result_str,
        membership_vec=membership,
    )

    return QueryResponse(
        query=request.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result_str,
        dominant_cluster=dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStats,
         summary="Cache statistics")
async def cache_stats() -> CacheStats:
    """Return current cache state including hit rate and bucket distribution."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    return CacheStats(**_cache.stats())


@app.delete("/cache", response_model=FlushResponse,
            summary="Flush the semantic cache")
async def flush_cache() -> FlushResponse:
    """Clear all cached entries and reset hit/miss counters."""
    if _cache is None:
        raise HTTPException(status_code=503, detail="Cache not initialised")
    _cache.flush()
    return FlushResponse(message="Medical Q&A cache flushed. All entries and stats reset.")


@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the frontend UI."""
    ui_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(ui_path)

@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "service": "Medical Q&A Semantic Search",
        "threshold": SIMILARITY_THRESHOLD,
        "n_clusters": N_CLUSTERS,
        "cache_ready": _cache is not None,
        "engine_ready": _engine is not None,
    }
