# BGE Embedding Demo

A hands-on demo of [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding) (BGE models by BAAI), covering the three core use cases of modern dense retrieval pipelines.

## What's Inside

| Demo | Description |
|------|-------------|
| **1. Basic Embeddings** | Encode sentences → compute cosine similarity matrix |
| **2. Semantic Search** | Encode a corpus + queries → retrieve Top-K passages |
| **3. Reranker** | Re-score candidates with a cross-encoder for higher precision |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all demos (models download automatically on first run)
python demo.py
```

> First run downloads `bge-small-en-v1.5` (~130 MB) and `bge-reranker-base` (~280 MB) from Hugging Face.

## Sample Output

```
Demo 1: Basic Embeddings & Cosine Similarity
────────────────────────────────────────────
Embedding dim: (4, 384)

Cosine similarity matrix:
          S0     S1     S2     S3
  S0   1.000  0.873  0.742  0.521
  S1   0.873  1.000  0.768  0.534
  S2   0.742  0.768  1.000  0.589
  S3   0.521  0.534  0.589  1.000

Demo 2: Semantic Search (Top-3)
────────────────────────────────
Query: "What is BGE embedding?"
  #1 [0.8821] BGE is a text embedding model developed by BAAI.
  #2 [0.8134] FlagEmbedding supports both English and multilingual models.
  #3 [0.7652] Semantic search finds results based on meaning rather than keywords.

Demo 3: Reranker
────────────────
Query: "What are the benefits of using BGE embeddings?"
  #1 [9.32]  BGE embeddings achieve state-of-the-art performance on retrieval benchmarks.
  #2 [8.11]  FlagEmbedding provides easy-to-use APIs for dense retrieval tasks.
  #3 [6.74]  BGE models support batch encoding for efficient large-scale indexing.
```

## Models

| Model | Size | Notes |
|-------|------|-------|
| `BAAI/bge-small-en-v1.5` | ~130 MB | Default — fast, CPU-friendly |
| `BAAI/bge-base-en-v1.5` | ~440 MB | Better quality |
| `BAAI/bge-large-en-v1.5` | ~1.3 GB | Best quality (English) |
| `BAAI/bge-m3` | ~2.3 GB | Multilingual (100+ languages, incl. Chinese) |
| `BAAI/bge-reranker-base` | ~280 MB | Default reranker |

Switch models by editing the constants at the top of `demo.py`:

```python
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
USE_FP16        = True   # Enable on GPU for ~2× speedup
```

## How It Works

```
Query  ──encode_queries()──▶  query_vec   ─┐
                                            ├── dot product ──▶ similarity scores
Corpus ──encode_corpus()──▶  corpus_vecs ─┘

                                   ┌── First-stage retrieval (bi-encoder)
Top-K candidates ──────────────────┤
                                   └── Second-stage reranking (cross-encoder)
```

**Bi-encoder** (FlagModel): encodes query and passages independently → fast retrieval over large corpora.

**Cross-encoder** (FlagReranker): scores (query, passage) pairs jointly → higher precision, used to re-rank Top-K results.

## Requirements

- Python 3.9+
- PyTorch 2.0+
- FlagEmbedding 1.2+

```
FlagEmbedding>=1.2.0
torch>=2.0.0
numpy>=1.24.0
```

## References

- [FlagEmbedding GitHub](https://github.com/FlagOpen/FlagEmbedding)
- [BGE Models on Hugging Face](https://huggingface.co/BAAI)
- [C-MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
