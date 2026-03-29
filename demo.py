"""
FlagEmbedding (BGE) Demo
========================
演示三种核心用法：
1. 基础嵌入 + 语义相似度
2. 语义搜索（检索最相关段落）
3. 重排序（Reranker 对候选结果精排）

模型说明：
- bge-small-en-v1.5  : 轻量英文模型（推荐本地快速测试）
- bge-base-en-v1.5   : 中等英文模型
- bge-large-en-v1.5  : 大型英文模型
- bge-m3             : 多语言模型（支持中文等 100+ 语言）
"""

import numpy as np
from FlagEmbedding import FlagModel, FlagReranker

# ─── 配置 ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"   # 小模型，下载快
RERANKER_MODEL  = "BAAI/bge-reranker-base"    # 对应的重排序模型
USE_FP16        = False                        # Mac CPU 下设为 False；GPU 可设 True

# BGE 英文检索时需要在 query 前加 instruction（passage 不加）
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# ─── 1. 基础嵌入 + 余弦相似度 ────────────────────────────────────────────────
def demo_basic_embeddings():
    print("\n" + "="*60)
    print("Demo 1: 基础嵌入 & 相似度计算")
    print("="*60)

    model = FlagModel(
        EMBEDDING_MODEL,
        query_instruction_for_retrieval=QUERY_INSTRUCTION,
        use_fp16=USE_FP16,
    )

    sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with many layers.",
        "Python is a popular programming language for data science.",
        "The weather today is sunny and warm.",
    ]

    # encode_corpus 不加 instruction；encode / encode_queries 加 instruction
    embeddings = model.encode_corpus(sentences)
    print(f"\n嵌入维度: {embeddings.shape}")  # (4, dim)

    # 余弦相似度（向量已 L2 归一化，直接点积即可）
    sim_matrix = embeddings @ embeddings.T
    print("\n句子间余弦相似度矩阵:")
    labels = [f"S{i}" for i in range(len(sentences))]
    header = "       " + "  ".join(f"{l:>5}" for l in labels)
    print(header)
    for i, row in enumerate(sim_matrix):
        vals = "  ".join(f"{v:5.3f}" for v in row)
        print(f"  {labels[i]:>4}  {vals}")

    print("\n句子内容:")
    for i, s in enumerate(sentences):
        print(f"  S{i}: {s}")


# ─── 2. 语义检索 ──────────────────────────────────────────────────────────────
def demo_semantic_search():
    print("\n" + "="*60)
    print("Demo 2: 语义检索（Top-K 最相关段落）")
    print("="*60)

    model = FlagModel(
        EMBEDDING_MODEL,
        query_instruction_for_retrieval=QUERY_INSTRUCTION,
        use_fp16=USE_FP16,
    )

    corpus = [
        "BGE is a text embedding model developed by BAAI.",
        "FlagEmbedding supports both English and multilingual models.",
        "Vector databases store high-dimensional embeddings for fast retrieval.",
        "Rerankers improve retrieval quality by re-scoring candidate passages.",
        "The Great Wall of China is one of the world's greatest landmarks.",
        "Python's rich ecosystem makes it ideal for machine learning projects.",
        "Semantic search finds results based on meaning rather than keywords.",
        "Beijing is the capital city of China.",
    ]

    queries = [
        "What is BGE embedding?",
        "How does semantic search work?",
    ]

    # 编码语料库（不加 instruction）
    corpus_embeddings = model.encode_corpus(corpus)

    # 编码查询（加 instruction）
    query_embeddings = model.encode_queries(queries)

    TOP_K = 3
    for q_idx, query in enumerate(queries):
        scores = query_embeddings[q_idx] @ corpus_embeddings.T
        top_k_idx = np.argsort(scores)[::-1][:TOP_K]
        print(f"\nQuery: \"{query}\"")
        print(f"  Top-{TOP_K} 结果:")
        for rank, idx in enumerate(top_k_idx, 1):
            print(f"    #{rank}  [score={scores[idx]:.4f}]  {corpus[idx]}")


# ─── 3. 重排序 ────────────────────────────────────────────────────────────────
def demo_reranker():
    print("\n" + "="*60)
    print("Demo 3: 重排序（Reranker 精排候选结果）")
    print("="*60)

    reranker = FlagReranker(RERANKER_MODEL, use_fp16=USE_FP16)

    query = "What are the benefits of using BGE embeddings?"
    candidates = [
        "BGE embeddings achieve state-of-the-art performance on retrieval benchmarks.",
        "The sky is blue because of Rayleigh scattering.",
        "BGE models support batch encoding for efficient large-scale indexing.",
        "Pizza is a popular Italian dish.",
        "FlagEmbedding provides easy-to-use APIs for dense retrieval tasks.",
    ]

    # compute_score 接受 (query, passage) 对列表
    pairs = [[query, passage] for passage in candidates]
    scores = reranker.compute_score(pairs)

    ranked = sorted(zip(scores, candidates), reverse=True)

    print(f"\nQuery: \"{query}\"")
    print("\n  重排后结果（分数越高越相关）:")
    for rank, (score, passage) in enumerate(ranked, 1):
        print(f"    #{rank}  [score={score:.4f}]  {passage}")


# ─── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("FlagEmbedding (BGE) Demo")
    print(f"  嵌入模型 : {EMBEDDING_MODEL}")
    print(f"  重排模型 : {RERANKER_MODEL}")

    demo_basic_embeddings()
    demo_semantic_search()
    demo_reranker()

    print("\n" + "="*60)
    print("所有 Demo 运行完毕！")
