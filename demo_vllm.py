"""
BGE High-Performance Deployment with vLLM
==========================================
High-throughput BGE inference for GPU production environments.
vLLM uses PagedAttention + Continuous Batching to achieve 5-10x
higher throughput than native Transformers under high concurrency.

Requirements:
    pip install vllm>=0.6.0

Demo covers:
  1. Embedding (bi-encoder)  — bge-base-en-v1.5 via vLLM
  2. Semantic Search          — query + corpus encoding
  3. Reranker                 — bge-reranker-v2-m3 via vLLM score API
  4. Throughput Benchmark     — FlagEmbedding vs vLLM speed comparison

GPU recommendation:
  - NVIDIA A10 / A100 / H100 for production
  - RTX 3090 / 4090 for development
  - CPU inference also works but negates most speed gains
"""

from __future__ import annotations

import time
import numpy as np

# ─── Config ───────────────────────────────────────────────────────────────────
# vLLM works best with bge-base or bge-large (BERT-based, full support)
# bge-small also works but with smaller throughput gains
EMBED_MODEL  = "BAAI/bge-base-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"   # XLM-RoBERTa based, vLLM score API
DTYPE        = "float16"                     # GPU: float16 / float32; CPU: float32
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _sep(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ─── Demo 1: vLLM Embedding ───────────────────────────────────────────────────
def demo_vllm_embeddings() -> None:
    _sep("Demo 1 (vLLM): Basic Embeddings & Cosine Similarity")
    from vllm import LLM

    llm = LLM(model=EMBED_MODEL, task="embed", dtype=DTYPE)

    sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Python is a popular programming language for data science.",
        "Natural language processing enables computers to understand text.",
        "The weather today is sunny and warm.",
    ]

    outputs = llm.embed(sentences)
    embeddings = np.array([o.outputs.embedding for o in outputs])
    print(f"\nEmbedding shape: {embeddings.shape}")

    # Normalize manually — vLLM embed output is not guaranteed to be L2-normalized
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-9)

    sim = embeddings_normalized @ embeddings_normalized.T
    labels = [f"S{i}" for i in range(len(sentences))]
    print("\nCosine similarity matrix:")
    print("       " + "  ".join(f"{l:>5}" for l in labels))
    for i, row in enumerate(sim):
        print(f"  {labels[i]:>4}  " + "  ".join(f"{v:5.3f}" for v in row))

    print("\nSentences:")
    for i, s in enumerate(sentences):
        print(f"  S{i}: {s}")


# ─── Demo 2: vLLM Semantic Search ────────────────────────────────────────────
def demo_vllm_semantic_search() -> None:
    _sep("Demo 2 (vLLM): Semantic Search (Top-K Retrieval)")
    from vllm import LLM

    llm = LLM(model=EMBED_MODEL, task="embed", dtype=DTYPE)

    corpus = [
        "BGE is a series of text embedding models developed by BAAI.",
        "FlagEmbedding supports dense, sparse, and multi-vector retrieval.",
        "Vector databases store high-dimensional embeddings for fast retrieval.",
        "Rerankers improve retrieval quality by re-scoring candidate passages.",
        "The Great Wall of China stretches thousands of miles.",
        "Python's rich ecosystem makes it ideal for machine learning projects.",
        "Semantic search finds results based on meaning, not just keywords.",
        "FAISS is an efficient library for approximate nearest neighbor search.",
        "Transformer models use attention mechanisms to process sequences.",
        "Beijing is the capital city of the People's Republic of China.",
    ]

    queries = [
        "What is the BGE embedding model?",
        "How does semantic search differ from keyword search?",
        "What tools are used for vector similarity search?",
    ]

    # Encode corpus + queries in a single call for efficiency
    all_texts = corpus + [QUERY_INSTRUCTION + q for q in queries]
    all_outputs = llm.embed(all_texts)
    all_vecs = np.array([o.outputs.embedding for o in all_outputs])

    # L2 normalize
    all_vecs /= np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-9

    corpus_emb = all_vecs[:len(corpus)]
    query_emb  = all_vecs[len(corpus):]

    TOP_K = 3
    for query, q_vec in zip(queries, query_emb):
        scores  = q_vec @ corpus_emb.T
        top_idx = np.argsort(scores)[::-1][:TOP_K]
        print(f'\nQuery: "{query}"')
        for rank, idx in enumerate(top_idx, 1):
            print(f"  #{rank}  [{scores[idx]:.4f}]  {corpus[idx]}")


# ─── Demo 3: vLLM Reranker ────────────────────────────────────────────────────
def demo_vllm_reranker() -> None:
    _sep("Demo 3 (vLLM): Reranker (Cross-encoder Score API)")
    from vllm import LLM

    # bge-reranker-v2-m3 uses vLLM's score task for cross-encoder scoring
    llm = LLM(model=RERANK_MODEL, task="score", dtype=DTYPE)

    query = "What are the advantages of using BGE for information retrieval?"
    candidates = [
        "BGE achieves state-of-the-art performance on multiple retrieval benchmarks.",
        "The sky appears blue due to Rayleigh scattering of sunlight.",
        "BGE supports batch encoding for efficient large-scale corpus indexing.",
        "FlagEmbedding provides simple APIs for building dense retrieval pipelines.",
        "Pizza originated in Naples, Italy, in the 18th century.",
        "BGE-M3 unifies dense, sparse, and multi-vector retrieval in one model.",
    ]

    # LLM.score(text_1_list, text_2_list) — batch cross-encoder scoring
    outputs = llm.score([query] * len(candidates), candidates)
    scores  = [o.outputs.score for o in outputs]

    ranked = sorted(zip(scores, candidates), reverse=True)
    print(f'\nQuery: "{query}"\n')
    print("  Reranked results:")
    for rank, (score, passage) in enumerate(ranked, 1):
        marker = " ✓" if score > 0 else ""
        print(f"  #{rank}  [{score:8.4f}]{marker}  {passage}")


# ─── Demo 4: Throughput Benchmark ────────────────────────────────────────────
def demo_throughput_benchmark(n_texts: int = 200) -> None:
    _sep(f"Demo 4: Throughput Benchmark (n={n_texts})")

    texts = [
        f"This is sample sentence number {i} for benchmarking embedding throughput."
        for i in range(n_texts)
    ]

    # vLLM
    try:
        from vllm import LLM
        llm = LLM(model=EMBED_MODEL, task="embed", dtype=DTYPE)
        t0 = time.perf_counter()
        outputs = llm.embed(texts)
        vllm_time = time.perf_counter() - t0
        vllm_tps = n_texts / vllm_time
        print(f"\n  vLLM         : {vllm_time:.3f}s  ({vllm_tps:.0f} texts/s)")
    except Exception as e:
        print(f"\n  vLLM         : skipped ({e})")
        vllm_tps = 0

    # FlagEmbedding (baseline)
    try:
        from FlagEmbedding import FlagModel
        flag_model = FlagModel(EMBED_MODEL, use_fp16=False)
        t0 = time.perf_counter()
        _ = flag_model.encode_corpus(texts)
        flag_time = time.perf_counter() - t0
        flag_tps = n_texts / flag_time
        print(f"  FlagEmbedding: {flag_time:.3f}s  ({flag_tps:.0f} texts/s)")
        if vllm_tps > 0:
            speedup = vllm_tps / flag_tps
            print(f"\n  vLLM speedup: {speedup:.1f}x")
    except Exception as e:
        print(f"  FlagEmbedding: skipped ({e})")


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("BGE + vLLM High-Performance Demo")
    print(f"  Embedding model : {EMBED_MODEL}")
    print(f"  Reranker model  : {RERANK_MODEL}")
    print(f"  dtype           : {DTYPE}")
    print()
    print("  Note: vLLM requires NVIDIA GPU for maximum performance.")
    print("        CPU mode works but reduces throughput gains.")

    demo_vllm_embeddings()
    demo_vllm_semantic_search()
    demo_vllm_reranker()
    demo_throughput_benchmark(n_texts=200)

    print("\n" + "="*60)
    print("All vLLM demos complete!")
