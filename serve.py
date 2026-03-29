"""
BGE Embedding Server (FlagEmbedding)
=====================================
Lightweight FastAPI service for local development and small-scale deployment.
For high-concurrency production, use serve_vllm.py instead.

Usage:
    pip install fastapi uvicorn
    python serve.py

Endpoints:
    GET  /health           — health check
    POST /embed            — standard BGE encoding (bi-encoder)
    POST /embed/m3         — BGE-M3 multi-repr encoding (dense + sparse + colbert)
    POST /rerank           — precision reranking (cross-encoder)
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from FlagEmbedding import FlagModel, FlagReranker, BGEM3FlagModel

# ─── Config ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL  = os.getenv("RERANKER_MODEL",  "BAAI/bge-reranker-base")
M3_MODEL        = os.getenv("M3_MODEL",        "BAAI/bge-m3")
USE_FP16        = os.getenv("USE_FP16", "false").lower() == "true"
HOST            = os.getenv("HOST", "0.0.0.0")
PORT            = int(os.getenv("PORT", "8080"))
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

app = FastAPI(title="BGE Embedding API", version="1.0.0")

# ─── Lazy model loading (initialized on first request) ────────────────────────
_flag_model: FlagModel | None = None
_reranker:   FlagReranker | None = None
_m3_model:   BGEM3FlagModel | None = None


def get_flag_model() -> FlagModel:
    global _flag_model
    if _flag_model is None:
        _flag_model = FlagModel(
            EMBEDDING_MODEL,
            query_instruction_for_retrieval=QUERY_INSTRUCTION,
            use_fp16=USE_FP16,
        )
    return _flag_model


def get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=USE_FP16)
    return _reranker


def get_m3() -> BGEM3FlagModel:
    global _m3_model
    if _m3_model is None:
        _m3_model = BGEM3FlagModel(M3_MODEL, use_fp16=USE_FP16)
    return _m3_model


# ─── Request / Response Schema ───────────────────────────────────────────────
class EmbedRequest(BaseModel):
    texts: list[str]
    mode: Literal["query", "passage"] = "passage"


class M3EmbedRequest(BaseModel):
    texts: list[str]
    return_dense: bool = True
    return_sparse: bool = False
    return_colbert_vecs: bool = False


class RerankRequest(BaseModel):
    query: str
    passages: list[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "embedding": EMBEDDING_MODEL,
            "reranker":  RERANKER_MODEL,
            "m3":        M3_MODEL,
        },
    }


@app.post("/embed")
def embed(req: EmbedRequest):
    model = get_flag_model()
    vecs = (
        model.encode_queries(req.texts)
        if req.mode == "query"
        else model.encode_corpus(req.texts)
    )
    return {"embeddings": vecs.tolist(), "dim": vecs.shape[1]}


@app.post("/embed/m3")
def embed_m3(req: M3EmbedRequest):
    out = get_m3().encode(
        req.texts,
        return_dense=req.return_dense,
        return_sparse=req.return_sparse,
        return_colbert_vecs=req.return_colbert_vecs,
    )
    resp: dict = {}
    if req.return_dense:
        resp["dense"] = out["dense_vecs"].tolist()
    if req.return_sparse:
        resp["sparse"] = [
            {k: float(v) for k, v in w.items()}
            for w in out["lexical_weights"]
        ]
    if req.return_colbert_vecs:
        resp["colbert"] = [v.tolist() for v in out["colbert_vecs"]]
    return resp


@app.post("/rerank")
def rerank(req: RerankRequest):
    scores = get_reranker().compute_score([[req.query, p] for p in req.passages])
    results = sorted(
        [
            {"index": i, "passage": p, "score": float(s)}
            for i, (p, s) in enumerate(zip(req.passages, scores))
        ],
        key=lambda x: x["score"],
        reverse=True,
    )
    return {"query": req.query, "results": results}


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
