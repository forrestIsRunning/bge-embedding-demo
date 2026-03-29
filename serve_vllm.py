"""
BGE High-Performance Server — vLLM Backend
==========================================
生产级 BGE 推理服务，基于 vLLM 实现高吞吐量。
相比 serve.py (FlagEmbedding)，在 GPU 上可提供 5-10x 以上吞吐量。

Architecture:
  FastAPI (async) → vLLM LLMEngine (Continuous Batching + PagedAttention)

Usage:
    # Option A: 本文件（FastAPI + vLLM Python API）
    pip install vllm>=0.6.0 fastapi uvicorn
    python serve_vllm.py

    # Option B: vLLM 内置 OpenAI 兼容服务（最简单，推荐生产）
    python -m vllm.entrypoints.openai.api_server \\
        --model BAAI/bge-base-en-v1.5 \\
        --task embed \\
        --dtype float16 \\
        --port 8080

Endpoints:
    GET  /health           — 健康检查
    POST /embed            — 编码文本（query / passage 模式）
    POST /rerank           — 精排 (query, passage) 对
    GET  /v1/models        — 列出已加载模型

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

# ─── 配置 ─────────────────────────────────────────────────────────────────────
EMBED_MODEL  = os.getenv("EMBED_MODEL",  "BAAI/bge-base-en-v1.5")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
DTYPE        = os.getenv("DTYPE",        "float16")      # CPU 请改 "float32"
HOST         = os.getenv("HOST",         "0.0.0.0")
PORT         = int(os.getenv("PORT",     "8080"))

# vLLM 张量并行度（多卡时设为 GPU 数量）
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# ─── 全局模型实例 ──────────────────────────────────────────────────────────────
_embed_llm:  object = None   # vllm.LLM
_rerank_llm: object = None   # vllm.LLM


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时预热模型"""
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
    normalize: bool = True   # L2 归一化（余弦相似度计算必须开启）


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


# ─── 启动 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT, loop="uvloop")
