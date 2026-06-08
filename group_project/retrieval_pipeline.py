"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất:

    Query
      ├→ Semantic Search (Task 5)  ──┐
      │                               ├→ Merge (RRF) → Rerank → Results
      ├→ Lexical Search (Task 6)  ──┘
      │
      └→ Nếu best_score < threshold → Fallback: PageIndex (Task 8)

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả bằng RRF (Reciprocal Rank Fusion)
    3. Rerank (mặc định RRF, optional cross-encoder)
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

import sys
from pathlib import Path

# Thêm project root vào path để import các module khác
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf, rerank
from src.task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

# Ngưỡng score tối thiểu cho hybrid results
# Nếu best result < threshold → kết quả không đủ tốt → fallback PageIndex
SCORE_THRESHOLD = 0.3

# Số lượng kết quả mặc định
DEFAULT_TOP_K = 5

# Phương pháp reranking: "rrf" (mặc định, không cần API)
# Đổi sang "cross_encoder" nếu có JINA_API_KEY
RERANK_METHOD = "rrf"


# =============================================================================
# MAIN RETRIEVAL FUNCTION
# =============================================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    verbose: bool = False,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline flow:
        1. Semantic search → top_k*2 dense results (ngữ nghĩa)
        2. Lexical search  → top_k*2 sparse results (từ khóa)
        3. Merge bằng RRF  → combined ranking
        4. Rerank (optional) → refined ranking
        5. Check threshold  → fallback PageIndex nếu cần

    Args:
        query: Câu truy vấn (tiếng Việt)
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng tối thiểu (dưới → fallback)
        use_reranking: Có áp dụng reranking không
        verbose: In chi tiết quá trình

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Score sau reranking
            'metadata': dict,    # source, type, chunk_index
            'source': str        # 'hybrid' hoặc 'pageindex'
        }
    """
    if verbose:
        print(f"  🔍 Query: {query}")

    # =========================================================================
    # Step 1: Chạy Semantic Search (dense retrieval)
    # Tìm chunks có ngữ nghĩa gần query (cosine similarity)
    # =========================================================================
    dense_results = semantic_search(query, top_k=top_k * 2)
    if verbose:
        print(f"  📊 Semantic: {len(dense_results)} results")

    # =========================================================================
    # Step 2: Chạy Lexical Search (sparse retrieval / BM25)
    # Tìm chunks chứa từ khóa giống query (BM25 scoring)
    # =========================================================================
    sparse_results = lexical_search(query, top_k=top_k * 2)
    if verbose:
        print(f"  📊 Lexical:  {len(sparse_results)} results")

    # =========================================================================
    # Step 3: Merge bằng RRF (Reciprocal Rank Fusion)
    # Kết hợp 2 ranked lists thành 1 — document xuất hiện trong cả 2 lists
    # sẽ được rank cao hơn
    # =========================================================================
    merged = rerank_rrf(
        ranked_lists=[dense_results, sparse_results],
        top_k=top_k * 2,  # Lấy nhiều hơn top_k để rerank tiếp
    )

    # Đánh dấu source = 'hybrid' (kết hợp semantic + lexical)
    for item in merged:
        item["source"] = "hybrid"

    if verbose:
        print(f"  📊 Merged (RRF): {len(merged)} results")

    # =========================================================================
    # Step 4: Rerank (optional) — refine ranking bằng phương pháp nâng cao
    # =========================================================================
    if use_reranking and merged:
        final_results = rerank(
            query, merged, top_k=top_k, method=RERANK_METHOD
        )
    else:
        final_results = merged[:top_k]

    if verbose:
        print(f"  📊 After rerank: {len(final_results)} results")

    # =========================================================================
    # Step 5: Check threshold → Fallback sang PageIndex
    # Nếu best score quá thấp → hybrid search không tốt → thử PageIndex
    # =========================================================================
    best_score = final_results[0]["score"] if final_results else 0

    if not final_results or best_score < score_threshold:
        if verbose:
            print(f"  ⚠ Best score ({best_score:.3f}) < threshold ({score_threshold})")
            print(f"  🔄 Fallback → PageIndex Vectorless...")

        fallback = pageindex_search(query, top_k=top_k)

        if fallback:
            return fallback
        elif verbose:
            print(f"  ⚠ PageIndex cũng không có kết quả. Trả về hybrid results.")

    # Đảm bảo mỗi item có 'source' field
    for item in final_results:
        if "source" not in item:
            item["source"] = "hybrid"

    return final_results[:top_k]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {q}")
        print("-" * 70)
        results = retrieve(q, top_k=3, verbose=True)
        print()
        for i, r in enumerate(results, 1):
            source = r.get("metadata", {}).get("source", "?")
            via = r.get("source", "?")
            print(f"  {i}. [{r['score']:.4f}] [{via}] ({source})")
            print(f"     {r['content'][:100]}...")

        if not results:
            print("  (Không có kết quả — chạy Task 3 & 4 trước)")
