# BGE Embedding Demo

A hands-on demo of [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) (BGE models by BAAI), covering the full retrieval pipeline from basic embeddings to production-grade vLLM deployment.

## What's Inside

| File | Description |
|------|-------------|
| `demo.py` | 6 demos: embeddings, semantic search, reranker, BGE-M3, hybrid search, cross-language |
| `demo_vllm.py` | Same demos re-implemented with **vLLM** backend + throughput benchmark |
| `serve.py` | Lightweight **FastAPI** server (FlagEmbedding, for dev / CPU) |
| `serve_vllm.py` | High-performance **FastAPI** server (vLLM backend, for GPU production) |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all 6 demos locally
python demo.py
```

> First run downloads models from Hugging Face automatically.

---

## Demos

### `demo.py` — 6 Core Demos

| # | Demo | Model | Highlights |
|---|------|-------|-----------|
| 1 | Basic Embeddings | bge-small-en-v1.5 | Cosine similarity matrix |
| 2 | Semantic Search | bge-small-en-v1.5 | Bi-encoder Top-K retrieval |
| 3 | Reranker | bge-reranker-base | Cross-encoder precision ranking |
| 4 | BGE-M3 Multi-repr. | bge-m3 | Dense + Sparse + ColBERT vectors |
| 5 | Hybrid Search | bge-m3 | Dense + Sparse RRF fusion |
| 6 | Cross-language | bge-m3 | Chinese query → English corpus |

#### Sample output

```
Demo 2: Semantic Search
────────────────────────
Query: "What is the BGE embedding model?"
  #1 [0.8589] BGE is a series of text embedding models developed by BAAI.
  #2 [0.7134] FlagEmbedding supports dense, sparse, and multi-vector retrieval.
  #3 [0.6292] Vector databases store high-dimensional embeddings for fast retrieval.

Demo 5: Hybrid Search (Dense + Sparse RRF)
─────────────────────────────────────────
  #  Dense    Sparse   Hybrid   Passage
  #1  0.8214   0.0731   0.0315  Hybrid search merges dense and sparse signals.
  #2  0.7621   0.0412   0.0295  Reciprocal Rank Fusion combines rankings.
  #3  0.7103   0.0891   0.0289  Dense retrieval excels at semantic matching.

Demo 6: Cross-language Retrieval
─────────────────────────────────
  ZH: BGE模型支持多少种语言？
  EN: How many languages does BGE support?
  ↳  [0.8821] BGE-M3 supports over 100 languages for multilingual retrieval.
```

---

## Deployment

### Option 1 — Lightweight (FlagEmbedding, CPU/GPU)

```bash
pip install fastapi "uvicorn[standard]"
python serve.py   # starts on http://0.0.0.0:8080
```

```bash
# Encode passages
curl -s -X POST http://localhost:8080/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["BGE is great for semantic search."], "mode": "passage"}' | jq .

# Rerank candidates
curl -s -X POST http://localhost:8080/rerank \
  -H "Content-Type: application/json" \
  -d '{"query": "What is BGE?", "passages": ["BGE is an embedding model.", "The sky is blue."]}' | jq .
```

### Option 2 — High-Performance (vLLM, NVIDIA GPU)

```bash
pip install "vllm>=0.6.0"

# Option A: Custom FastAPI + vLLM backend (flexible, supports reranker)
python serve_vllm.py

# Option B: vLLM built-in OpenAI-compatible server (simplest for embedding)
python -m vllm.entrypoints.openai.api_server \
    --model BAAI/bge-base-en-v1.5 \
    --task embed \
    --dtype float16 \
    --port 8080
```

**OpenAI-compatible client:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="dummy")
response = client.embeddings.create(
    model="BAAI/bge-base-en-v1.5",
    input=["BGE is great!", "Semantic search rocks."],
)
vectors = [e.embedding for e in response.data]
```

### Throughput Comparison

| Backend | Hardware | Throughput |
|---------|----------|-----------|
| FlagEmbedding | CPU (M2) | ~200 texts/s |
| FlagEmbedding | A100 GPU | ~2,000 texts/s |
| vLLM | A100 GPU | ~8,000–15,000 texts/s |
| vLLM (FP8) | H100 GPU | ~20,000+ texts/s |

> Run `python demo_vllm.py` to benchmark on your hardware.

---

## Models

| Model | Size | Use case |
|-------|------|---------|
| `BAAI/bge-small-en-v1.5` | ~130 MB | Fast local testing |
| `BAAI/bge-base-en-v1.5` | ~440 MB | Recommended for vLLM |
| `BAAI/bge-large-en-v1.5` | ~1.3 GB | Best English quality |
| `BAAI/bge-m3` | ~2.3 GB | Multilingual + multi-repr. |
| `BAAI/bge-reranker-base` | ~280 MB | Lightweight reranker |
| `BAAI/bge-reranker-v2-m3` | ~570 MB | High-quality reranker (vLLM) |

Swap models by editing the constants at the top of any file:

```python
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
USE_FP16        = True   # Enable on GPU
```

---

## How It Works

```
Query ──encode_queries()──▶ query_vec ─┐
                                        ├── dot product ──▶ similarity scores
Corpus ──encode_corpus()──▶ corpus_vecs─┘

First-stage retrieval (bi-encoder, fast)
            │
            ▼ Top-K candidates
Second-stage reranking (cross-encoder, precise)
            │
            ▼ Final ranked results

BGE-M3 adds: sparse lexical weights + ColBERT multi-vector
Hybrid search: RRF fusion of dense + sparse rankings
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- FlagEmbedding 1.2+
- vLLM 0.6+ (optional, GPU only)

---

## References

- [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [BGE Models on Hugging Face](https://huggingface.co/BAAI)
- [vLLM Documentation](https://docs.vllm.ai)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
