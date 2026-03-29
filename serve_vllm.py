"""
BGE High-Performance Server — vLLM Backend
==========================================
Production-grade BGE inference service powered by vLLM for high throughput.
Compared to serve.py (FlagEmbedding), delivers 5-10x higher throughput on GPU.

Architecture:
  FastAPI (async) → vLLM LLMEngine (Continuous Batching + PagedAttention)

Usage:
    # Option A: this file (FastAPI + vLLM Python API)
    pip install vllm>=0.6.0 fastapi uvicorn
    python serve_vllm.py

    # Option B: vLLM built-in OpenAI-compatible server (simplest, recommended for production)
    python -m vllm.entrypoints.openai.api_server \\
        --model BAAI/bge-base-en-v1.5 \\
        --task embed \\
        --dtype float16 \\
        --port 8080

Endpoints:
    GET  /health           — health check
    POST /embed            — encode text (query / passage mode)
    POST /rerank           — rerank (query, passage) pairs
    GET  /v1/models        — list loaded models

OpenAI-compatible client example:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
    resp = client.embeddings.create(model="BAAI/bge-base-en-v1.5", input=texts)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── Config ───────────────────────────────────────────────────────────────────
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DTYPE        = os.getenv("DTYPE",        "float16")      # Use "float32" for CPU
HOST         = os.getenv("HOST",         "0.0.0.0")
PORT         = int(os.getenv("PORT",     "8080"))

# vLLM tensor parallel size (set to number of GPUs for multi-GPU)
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ─── Global model instances ───────────────────────────────────────────────────
_embed_llm:  object = None   # vllm.LLM
_rerank_llm: object = None   # vllm.LLM


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up models at application startup."""
    global _embed_llm, _rerank_llm
    from vllm import LLM

    print(f"[startup] Loading embedding model: {EMBED_MODEL}")
    _embed_llm = LLM(
        model=EMBED_MODEL,
        task="embed",
        dtype=DTYPE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
    )

    print(f"[startup] Loading reranker model:  {RERANK_MODEL}")
    _rerank_llm = LLM(
        model=RERANK_MODEL,
        task="score",
        dtype=DTYPE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True,
    )
    print("[startup] Models ready.")
    yield
    print("[shutdown] Cleaning up.")


app = FastAPI(
    title="BGE High-Performance API (vLLM)",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Schema ───────────────────────────────────────────────────────────────────
class EmbedRequest(BaseModel):
    texts: list[str]
    mode: Literal["query", "passage"] = "passage"
    normalize: bool = True   # L2 normalize (required for cosine similarity)


class RerankRequest(BaseModel):
    query: str
    passages: list[str]
    top_k: int | None = None


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "vllm",
        "models": {"embedding": EMBED_MODEL, "reranker": RERANK_MODEL},
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": EMBED_MODEL,  "object": "model", "owned_by": "baai"},
            {"id": RERANK_MODEL, "object": "model", "owned_by": "baai"},
        ],
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    if _embed_llm is None:
        raise HTTPException(503, "Embedding model not ready")

    texts = (
        [QUERY_INSTRUCTION + t for t in req.texts]
        if req.mode == "query"
        else req.texts
    )

    t0 = time.perf_counter()
    outputs = _embed_llm.embed(texts)
    latency = time.perf_counter() - t0

    vecs = np.array([o.outputs.embedding for o in outputs], dtype=np.float32)

    if req.normalize:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / (norms + 1e-9)

    return {
        "embeddings": vecs.tolist(),
        "dim":        int(vecs.shape[1]),
        "count":      len(req.texts),
        "latency_ms": round(latency * 1000, 2),
    }


@app.post("/rerank")
def rerank(req: RerankRequest):
    if _rerank_llm is None:
        raise HTTPException(503, "Reranker model not ready")

    n = len(req.passages)
    t0 = time.perf_counter()
    outputs = _rerank_llm.score([req.query] * n, req.passages)
    latency = time.perf_counter() - t0

    scored = [
        {"index": i, "passage": p, "score": float(o.outputs.score)}
        for i, (p, o) in enumerate(zip(req.passages, outputs))
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)

    if req.top_k is not None:
        scored = scored[: req.top_k]

    return {
        "query":      req.query,
        "results":    scored,
        "latency_ms": round(latency * 1000, 2),
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, loop="uvloop")
