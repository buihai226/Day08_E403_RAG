"""
Task 7 — Reranking Module.

Implement 3 phương pháp reranking:
    1. RRF (Reciprocal Rank Fusion): gộp kết quả từ nhiều ranker
    2. MMR (Maximal Marginal Relevance): cân bằng relevance + diversity
    3. Cross-encoder: dùng Jina Reranker API (cần API key)

Phương pháp mặc định: RRF — không cần API key, không cần GPU.
"""

import os
import numpy as np
from typing import Optional


# =============================================================================
# UTILITY: Cosine Similarity
# =============================================================================

def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Tính cosine similarity giữa 2 vectors.
    cos(a, b) = (a · b) / (||a|| × ||b||)
    Kết quả ∈ [-1, 1], trong đó 1 = giống hoàn toàn.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


# =============================================================================
# RRF — Reciprocal Rank Fusion
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker thành 1 list duy nhất.

    Paper: Cormack et al. (2009) "Reciprocal Rank Fusion outperforms
           Condorcet and individual Rank Learning Methods"

    Formula: RRF_score(d) = Σ_{r ∈ rankers} 1 / (k + rank_r(d))

    Ý tưởng: Mỗi ranker "bỏ phiếu" cho document. Document xuất hiện ở
    rank cao trong nhiều ranker → score cao. Parameter k=60 (từ paper)
    giúp smooth: tránh rank 1 quá áp đảo rank 2.

    Ví dụ: document xuất hiện ở rank 1 trong semantic + rank 3 trong BM25:
        RRF = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60 từ paper gốc)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}   # content → tổng RRF score
    content_map = {}  # content → full dict (để trả về)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):  # rank bắt đầu từ 1
            key = item["content"]  # Dùng content làm key unique

            # Cộng dồn RRF score từ mỗi ranker
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank)

            # Lưu item gốc (lấy cái mới nhất nếu trùng)
            if key not in content_map:
                content_map[key] = item

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Build output
    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)  # Override score bằng RRF score
        results.append(item)

    return results


# =============================================================================
# MMR — Maximal Marginal Relevance
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Paper: Carbonell & Goldstein (1998)

    Formula: MMR = λ × sim(query, doc) - (1-λ) × max(sim(doc, selected_docs))

    Ý tưởng: Không chỉ chọn document giống query nhất, mà còn ĐA DẠNG.
    Nếu 2 chunks nói cùng 1 nội dung, chỉ lấy 1 → giảm redundancy.

    λ = 0.7: ưu tiên relevance (70%) hơn diversity (30%)
    λ = 1.0: chỉ quan tâm relevance (= pure cosine similarity search)
    λ = 0.0: chỉ quan tâm diversity (= chọn documents khác nhau nhất)

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content', 'score', 'embedding', 'metadata'}
        top_k: Số lượng kết quả
        lambda_param: Trade-off relevance (1.0) vs diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if not candidates:
        return []

    # Nếu candidate không có embedding, return theo score gốc
    if "embedding" not in candidates[0]:
        return candidates[:top_k]

    selected = []       # Indices đã chọn
    remaining = list(range(len(candidates)))  # Indices còn lại

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_mmr_score = float('-inf')

        for idx in remaining:
            # Relevance: similarity giữa query và candidate
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Diversity: max similarity với các candidates đã chọn
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(
                    candidates[idx]["embedding"],
                    candidates[sel_idx]["embedding"]
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score = relevance - redundancy
            mmr_score = (lambda_param * relevance
                         - (1 - lambda_param) * max_sim_to_selected)

            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    # Build output, giữ nguyên score gốc + thêm mmr_rank
    results = []
    for rank, idx in enumerate(selected):
        item = candidates[idx].copy()
        # Xóa embedding khỏi output (quá lớn, không cần trả về)
        item.pop("embedding", None)
        results.append(item)

    return results


# =============================================================================
# CROSS-ENCODER RERANKING (Jina Reranker API)
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank bằng cross-encoder model (Jina Reranker v2 multilingual).

    Cross-encoder khác bi-encoder (embedding search):
        - Bi-encoder: embed query & doc riêng, so cosine → NHANH nhưng kém chính xác
        - Cross-encoder: nhập cặp (query, doc) vào model → score → CHẬM nhưng CHÍNH XÁC

    Jina Reranker v2 hỗ trợ tiếng Việt (multilingual).

    Cần: JINA_API_KEY trong .env (miễn phí 1M tokens/tháng)
    Đăng ký tại: https://jina.ai/reranker/

    Args:
        query: Câu truy vấn
        candidates: List of {'content', 'score', 'metadata'}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored by cross-encoder.
    """
    import requests

    api_key = os.getenv("JINA_API_KEY", "")
    if not api_key:
        print("  ⚠ JINA_API_KEY chưa set. Dùng RRF thay thế.")
        # Fallback: trả về candidates theo score gốc
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    # Gọi Jina Reranker API
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()

    reranked = response.json()["results"]
    results = []
    for r in reranked:
        item = candidates[r["index"]].copy()
        item["score"] = round(r["relevance_score"], 4)
        results.append(item)

    return results


# =============================================================================
# UNIFIED RERANK INTERFACE
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # Mặc định RRF — không cần API key
    **kwargs,
) -> list[dict]:
    """
    Unified reranking interface — chọn method phù hợp.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: "rrf" | "mmr" | "cross_encoder"
        **kwargs: Extra params (lambda_param cho MMR, ranked_lists cho RRF)

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        # MMR cần query_embedding
        query_embedding = kwargs.get("query_embedding")
        if query_embedding is None:
            # Auto-embed query nếu chưa có
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            query_embedding = model.encode(query).tolist()
        lambda_param = kwargs.get("lambda_param", 0.7)
        return rerank_mmr(query_embedding, candidates, top_k, lambda_param)

    elif method == "rrf":
        # RRF cần ranked_lists — nếu chỉ có 1 list, wrap nó
        ranked_lists = kwargs.get("ranked_lists")
        if ranked_lists is None:
            ranked_lists = [candidates]
        return rerank_rrf(ranked_lists, top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test RRF với dummy data
    print("=== Test RRF ===")
    list1 = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.9, "metadata": {"source": "BLHS"}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.7, "metadata": {"source": "BLHS"}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.5, "metadata": {"source": "news"}},
    ]
    list2 = [
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 5.2, "metadata": {"source": "BLHS"}},
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 4.1, "metadata": {"source": "BLHS"}},
        {"content": "Quy trình cai nghiện bắt buộc", "score": 2.3, "metadata": {"source": "luật"}},
    ]

    results = rerank_rrf([list1, list2], top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}")

    print("\n=== Test MMR ===")
    # MMR cần embeddings — dùng random vectors cho demo
    candidates_mmr = [
        {"content": "Chunk A", "score": 0.9, "embedding": [1, 0, 0], "metadata": {}},
        {"content": "Chunk B (giống A)", "score": 0.85, "embedding": [0.99, 0.1, 0], "metadata": {}},
        {"content": "Chunk C (khác)", "score": 0.7, "embedding": [0, 1, 0], "metadata": {}},
    ]
    results = rerank_mmr([1, 0, 0], candidates_mmr, top_k=2, lambda_param=0.5)
    for r in results:
        print(f"  [{r['score']:.3f}] {r['content']}")
