"""
Task 6 — Lexical Search Module (BM25).

BM25 (Best Matching 25) — thuật toán tìm kiếm từ khóa (lexical/sparse):
    - Dựa trên tần suất xuất hiện từ trong document (TF) và độ hiếm (IDF)
    - KHÔNG hiểu ngữ nghĩa — chỉ so khớp từ khóa chính xác
    - Tốt cho: tên riêng, số điều luật, mã văn bản (Điều 248, NĐ 105,...)

Formula: score(q,d) = Σ IDF(qi) × (tf(qi,d) × (k1+1)) / (tf(qi,d) + k1×(1-b+b×|d|/avgdl))
    - k1=1.5: điều chỉnh term saturation (từ lặp nhiều → bão hòa dần)
    - b=0.75: chuẩn hóa theo độ dài document (doc dài không bị ưu tiên quá)

# =============================================================================
# BONUS: SO SÁNH BM25 VỚI TF-IDF (+5 điểm)
# =============================================================================
#
# TF-IDF (Term Frequency - Inverse Document Frequency):
#   - TF(t,d) = số lần từ t xuất hiện trong document d / tổng từ trong d
#   - IDF(t) = log(N / df(t))  — N: tổng documents, df: số doc chứa từ t
#   - Score = TF × IDF
#
# BM25 CẢI TIẾN SO VỚI TF-IDF:
#   1. Term Saturation: TF-IDF tăng tuyến tính theo TF → từ lặp 100 lần
#      được gấp 10 lần so với 10 lần. BM25 dùng k1 để bão hòa: sau ~5 lần
#      thì tăng thêm rất ít → thực tế hơn.
#   2. Document Length Normalization: TF-IDF không chuẩn hóa độ dài.
#      BM25 dùng parameter b để giảm điểm cho document dài (vì chứa nhiều
#      từ tự nhiên, không nhất thiết relevant hơn).
#   3. IDF smoothing: BM25 dùng log((N-df+0.5)/(df+0.5)) thay vì log(N/df)
#      → tránh IDF=0 khi từ xuất hiện ở mọi document.
#
# KHI NÀO DÙNG CÁI NÀO?
#   - TF-IDF: đơn giản, tốt cho document classification, feature extraction
#   - BM25: tốt hơn cho search/retrieval, là tiêu chuẩn công nghiệp
#     (Elasticsearch, Lucene đều dùng BM25)
# =============================================================================

Cài đặt:
    pip install rank-bm25 numpy
"""

import json
from pathlib import Path
import numpy as np

# Đường dẫn tới file chunks đã lưu ở Task 4
CHUNKS_JSON = Path(__file__).parent.parent / "data" / "chunks.json"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chromadb"
COLLECTION_NAME = "drug_law_docs"

# Cache BM25 index và corpus
_bm25 = None
_corpus: list[dict] = []

# Cache TF-IDF (bonus: lexical search khác BM25)
_tfidf_vectorizer = None
_tfidf_matrix = None


def _load_corpus() -> list[dict]:
    """
    Load corpus chunks từ file JSON (nhanh hơn query ChromaDB).
    File chunks.json được tạo bởi Task 4.
    Fallback: load trực tiếp từ ChromaDB nếu không có JSON.
    """
    # Ưu tiên đọc từ JSON (nhanh)
    if CHUNKS_JSON.exists():
        data = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
        return data

    # Fallback: đọc từ ChromaDB
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection = client.get_collection(name=COLLECTION_NAME)
        results = collection.get(include=["documents", "metadatas"])
        corpus = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            corpus.append({"content": doc, "metadata": meta})
        return corpus
    except Exception as e:
        print(f"  ⚠ Không thể load corpus: {e}")
        print("    Chạy Task 4 trước để tạo chunks.")
        return []


def _build_index():
    """Build BM25 index từ corpus (lazy, chỉ build 1 lần)."""
    global _bm25, _corpus
    if _bm25 is not None:
        return

    from rank_bm25 import BM25Okapi

    _corpus = _load_corpus()
    if not _corpus:
        return

    # Tokenize — cho tiếng Việt, đơn giản dùng split()
    # (Nâng cao: dùng underthesea.word_tokenize để tách từ ghép)
    tokenized_corpus = [doc["content"].lower().split() for doc in _corpus]

    # Build BM25 index với default params: k1=1.5, b=0.75
    _bm25 = BM25Okapi(tokenized_corpus)
    print(f"  📚 BM25 index built: {len(_corpus)} documents")


def lexical_search(query: str, top_k: int = 10, method: str = "bm25") -> list[dict]:
    """
    Tìm kiếm từ khóa (lexical / sparse retrieval).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        method: "bm25" (mặc định) hoặc "tfidf"

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    if method == "tfidf":
        return _tfidf_search(query, top_k)
    return _bm25_search(query, top_k)


def _bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm bằng BM25 (tiêu chuẩn công nghiệp cho search/retrieval)."""
    _build_index()

    if _bm25 is None or not _corpus:
        return []

    # Tokenize query (cùng cách với corpus)
    tokenized_query = query.lower().split()

    # Tính BM25 score cho mỗi document
    scores = _bm25.get_scores(tokenized_query)

    # Lấy top_k indices có score cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Build kết quả, chỉ lấy những doc có score > 0
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": round(float(scores[idx]), 4),
                "metadata": _corpus[idx]["metadata"],
            })

    return results


# =============================================================================
# BONUS: TF-IDF LEXICAL SEARCH (phương pháp khác BM25)
#
# Khác biệt cốt lõi so với BM25 (đã giải thích chi tiết ở đầu file):
#   - TF-IDF: score = TF × IDF, tăng TUYẾN TÍNH theo tần suất từ, KHÔNG chuẩn
#     hóa độ dài document → document dài dễ được điểm cao bất hợp lý.
#   - BM25: thêm term saturation (k1) + document length normalization (b) →
#     thực tế hơn cho retrieval.
# Ở đây dùng cosine similarity giữa vector TF-IDF của query và của từng chunk.
# =============================================================================

def _build_tfidf():
    """Build TF-IDF matrix từ corpus (lazy)."""
    global _tfidf_vectorizer, _tfidf_matrix, _corpus
    if _tfidf_matrix is not None:
        return

    from sklearn.feature_extraction.text import TfidfVectorizer

    if not _corpus:
        _corpus = _load_corpus()
    if not _corpus:
        return

    texts = [doc["content"].lower() for doc in _corpus]
    # token_pattern giữ từ có dấu tiếng Việt (\w trong unicode)
    _tfidf_vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(texts)
    print(f"  📚 TF-IDF matrix built: {_tfidf_matrix.shape[0]} docs "
          f"× {_tfidf_matrix.shape[1]} terms")


def _tfidf_search(query: str, top_k: int = 10) -> list[dict]:
    """Tìm kiếm bằng TF-IDF + cosine similarity."""
    from sklearn.metrics.pairwise import cosine_similarity

    _build_tfidf()
    if _tfidf_matrix is None or not _corpus:
        return []

    query_vec = _tfidf_vectorizer.transform([query.lower()])
    scores = cosine_similarity(query_vec, _tfidf_matrix)[0]

    top_indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": _corpus[idx]["content"],
                "score": round(float(scores[idx]), 4),
                "metadata": _corpus[idx]["metadata"],
            })
    return results


if __name__ == "__main__":
    # Test với 3 câu hỏi mẫu
    test_queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "nghệ sĩ bị bắt ma túy",
        "cai nghiện bắt buộc",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("-" * 60)
        results = lexical_search(q, top_k=5)
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "?")
            print(f"  {i}. [{r['score']:.3f}] ({source}) {r['content'][:80]}...")
        if not results:
            print("  (Không có kết quả — chạy Task 4 trước)")
