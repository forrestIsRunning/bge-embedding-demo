"""
FlagEmbedding (BGE) Demo
========================
6 demos covering the full retrieval pipeline from basics to advanced:

  Demo 1. Basic Embeddings      — encode sentences + cosine similarity matrix
  Demo 2. Semantic Search       — Top-K retrieval with bi-encoder
  Demo 3. Reranker              — re-rank candidates with cross-encoder
  Demo 4. BGE-M3 Multi-repr.   — dense / sparse / ColBERT multi-representation
  Demo 5. Hybrid Search         — dense + sparse RRF fusion
  Demo 6. Cross-language        — Chinese query → English corpus cross-lingual retrieval

Models:
  bge-small-en-v1.5  lightweight English model (Demo 1-3, fast local testing)
  bge-reranker-base  English reranker
  bge-m3             multilingual multi-repr model (Demo 4-6, 100+ languages)
"""

import numpy as np
from FlagEmbedding import FlagModel, FlagReranker, BGEM3FlagModel

# ─── Config ───────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL    = "BAAI/bge-reranker-base"
M3_MODEL          = "BAAI/bge-m3"
USE_FP16          = False   # Set True on GPU, keep False on CPU
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _sep(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# ─── Demo 1: Basic Embeddings + Cosine Similarity ────────────────────────────
def demo_basic_embeddings(model: FlagModel) -> None:
    _sep("Demo 1: Basic Embeddings & Cosine Similarity")

    sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Python is a popular programming language for data science.",
        "Natural language processing enables computers to understand text.",
        "The weather today is sunny and warm.",
    ]

    embeddings = model.encode_corpus(sentences)
    print(f"\nEmbedding shape: {embeddings.shape}")

    sim = embeddings @ embeddings.T
    labels = [f"S{i}" for i in range(len(sentences))]
    print("\nCosine similarity matrix:")
    print("       " + "  ".join(f"{l:>5}" for l in labels))
    for i, row in enumerate(sim):
        print(f"  {labels[i]:>4}  " + "  ".join(f"{v:5.3f}" for v in row))

    print("\nSentences:")
    for i, s in enumerate(sentences):
        print(f"  S{i}: {s}")


# ─── Demo 2: Semantic Search ─────────────────────────────────────────────────
def demo_semantic_search(model: FlagModel) -> None:
    _sep("Demo 2: Semantic Search (Bi-encoder Top-K Retrieval)")

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

    corpus_emb = model.encode_corpus(corpus)
    query_emb  = model.encode_queries(queries)

    TOP_K = 3
    for query, q_vec in zip(queries, query_emb):
        scores  = q_vec @ corpus_emb.T
        top_idx = np.argsort(scores)[::-1][:TOP_K]
        print(f'\nQuery: "{query}"')
        for rank, idx in enumerate(top_idx, 1):
            print(f"  #{rank}  [{scores[idx]:.4f}]  {corpus[idx]}")


# ─── Demo 3: Reranker ────────────────────────────────────────────────────────
def demo_reranker() -> None:
    _sep("Demo 3: Reranker (Cross-encoder Precision Ranking)")

    reranker = FlagReranker(RERANKER_MODEL, use_fp16=USE_FP16)

    query = "What are the advantages of using BGE for information retrieval?"
    candidates = [
        "BGE achieves state-of-the-art performance on multiple retrieval benchmarks.",
        "The sky appears blue due to Rayleigh scattering of sunlight.",
        "BGE supports batch encoding for efficient large-scale corpus indexing.",
        "FlagEmbedding provides simple APIs for building dense retrieval pipelines.",
        "Pizza originated in Naples, Italy, in the 18th century.",
        "BGE-M3 unifies dense, sparse, and multi-vector retrieval in one model.",
    ]

    scores = reranker.compute_score([[query, c] for c in candidates])
    ranked = sorted(zip(scores, candidates), reverse=True)

    print(f'\nQuery: "{query}"\n')
    print("  Reranked results (higher = more relevant):")
    for rank, (score, passage) in enumerate(ranked, 1):
        marker = " ✓" if score > 0 else ""
        print(f"  #{rank}  [{score:8.4f}]{marker}  {passage}")


# ─── Demo 4: BGE-M3 Multi-representation Retrieval ───────────────────────────
def demo_bge_m3(m3: BGEM3FlagModel) -> None:
    _sep("Demo 4: BGE-M3 — Dense / Sparse / Multi-vector (ColBERT)")

    sentences = [
        "BGE-M3 unifies dense, sparse, and multi-vector retrieval.",
        "Sparse retrieval matches keywords using lexical weights.",
        "ColBERT performs fine-grained token-level similarity matching.",
        "Hybrid retrieval combines multiple signals for better coverage.",
    ]

    out = m3.encode(
        sentences,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    dense   = out["dense_vecs"]       # shape: (N, 1024)
    sparse  = out["lexical_weights"]  # list of {token_id: weight}
    colbert = out["colbert_vecs"]     # list of (seq_len, 1024)

    print(f"\nDense vector shape  : {dense.shape}")
    print(f"ColBERT shapes      : {[v.shape for v in colbert]}")

    # Show top-5 tokens per sentence by sparse weight
    tokenizer = m3.tokenizer
    print("\nSparse top-5 tokens per sentence:")
    for i, weights in enumerate(sparse):
        top5 = sorted(weights.items(), key=lambda x: float(x[1]), reverse=True)[:5]
        tokens = [(tokenizer.decode([int(tid)]).strip(), round(float(w), 4)) for tid, w in top5]
        print(f"  S{i}: {tokens}")

    # Dense cosine similarity matrix
    sim = dense @ dense.T
    labels = [f"S{i}" for i in range(len(sentences))]
    print("\nDense cosine similarity:")
    print("       " + "  ".join(f"{l:>5}" for l in labels))
    for i, row in enumerate(sim):
        print(f"  {labels[i]:>4}  " + "  ".join(f"{v:5.3f}" for v in row))


# ─── Demo 5: Hybrid Search (RRF Fusion) ──────────────────────────────────────
def demo_hybrid_search(m3: BGEM3FlagModel) -> None:
    _sep("Demo 5: Hybrid Search (Dense + Sparse RRF Fusion)")

    corpus = [
        "BGE-M3 achieves top scores on MTEB multilingual benchmarks.",
        "Reciprocal Rank Fusion combines rankings from multiple retrievers.",
        "Dense retrieval excels at semantic matching across paraphrases.",
        "BM25 is a classic sparse retrieval algorithm based on term frequency.",
        "Hybrid search merges dense and sparse signals for better recall.",
        "Vector quantization reduces memory footprint of embedding indices.",
        "The Eiffel Tower is located in Paris, France.",
        "FAISS supports both exact and approximate nearest neighbor search.",
    ]
    query = "How to combine dense and sparse retrieval for better results?"

    out_c = m3.encode(corpus,  return_dense=True, return_sparse=True)
    out_q = m3.encode([query], return_dense=True, return_sparse=True)

    d_q = out_q["dense_vecs"][0]
    d_c = out_c["dense_vecs"]
    s_q = out_q["lexical_weights"][0]
    s_c = out_c["lexical_weights"]

    dense_scores = d_q @ d_c.T

    def sparse_dot(qw: dict, pw: dict) -> float:
        return sum(float(qw.get(k, 0)) * float(v) for k, v in pw.items())

    sparse_scores = np.array([sparse_dot(s_q, s) for s in s_c])

    def rrf(scores: np.ndarray, k: int = 60) -> np.ndarray:
        order = np.argsort(scores)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(order))
        return 1.0 / (k + ranks + 1)

    hybrid = rrf(dense_scores) + rrf(sparse_scores)

    print(f'\nQuery: "{query}"\n')
    print(f"  {'#':>2}  {'Dense':>7}  {'Sparse':>7}  {'Hybrid':>7}  Passage")
    print(f"  {'─'*2}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*50}")
    for rank, idx in enumerate(np.argsort(hybrid)[::-1][:5], 1):
        print(f"  #{rank}  {dense_scores[idx]:7.4f}  {sparse_scores[idx]:7.4f}  {hybrid[idx]:7.4f}  {corpus[idx]}")


# ─── Demo 6: Cross-language Retrieval ────────────────────────────────────────
def demo_cross_language(m3: BGEM3FlagModel) -> None:
    _sep("Demo 6: Cross-language Retrieval (Chinese Query → English Corpus)")

    corpus = [
        "BGE-M3 supports over 100 languages for multilingual retrieval.",
        "Machine learning models can process multilingual datasets.",
        "The Great Wall of China is a famous historical monument.",
        "Natural language processing enables cross-lingual understanding.",
        "Vector embeddings capture semantic meaning across different languages.",
        "Paris is the capital of France and home to the Eiffel Tower.",
        "Deep learning has revolutionized computer vision and NLP tasks.",
    ]

    queries = [
        ("BGE模型支持多少种语言？",      "How many languages does BGE support?"),
        ("什么是自然语言处理？",          "What is natural language processing?"),
        ("长城在哪里？",                  "Where is the Great Wall?"),
        ("深度学习改变了哪些领域？",      "What fields has deep learning changed?"),
    ]

    corpus_emb = m3.encode(corpus,                        return_dense=True)["dense_vecs"]
    query_emb  = m3.encode([q for q, _ in queries],       return_dense=True)["dense_vecs"]

    print("\n[Chinese Query → English Corpus]\n")
    for (zh, en), q_vec in zip(queries, query_emb):
        scores = q_vec @ corpus_emb.T
        top1   = int(np.argmax(scores))
        print(f"  ZH: {zh}")
        print(f"  EN: {en}")
        print(f"  ↳  [{scores[top1]:.4f}] {corpus[top1]}")
        print()


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FlagEmbedding (BGE) Demo")
    print(f"  Embedding : {EMBEDDING_MODEL}")
    print(f"  Reranker  : {RERANKER_MODEL}")
    print(f"  BGE-M3    : {M3_MODEL}")

    # Shared model instances — avoids loading models twice
    flag_model = FlagModel(
        EMBEDDING_MODEL,
        query_instruction_for_retrieval=QUERY_INSTRUCTION,
        use_fp16=USE_FP16,
    )
    m3_model = BGEM3FlagModel(M3_MODEL, use_fp16=USE_FP16)

    demo_basic_embeddings(flag_model)
    demo_semantic_search(flag_model)
    demo_reranker()
    demo_bge_m3(m3_model)
    demo_hybrid_search(m3_model)
    demo_cross_language(m3_model)

    print("\n" + "="*60)
    print("All demos complete!")
