"""
Task 5 — Semantic Search Module (Dense Retrieval).

Tìm kiếm ngữ nghĩa (semantic search) sử dụng vector similarity:
    1. Embed query bằng cùng model đã dùng ở Task 4 (all-MiniLM-L6-v2)
    2. Tìm top_k chunks gần nhất trong ChromaDB (cosine similarity)
    3. Trả về kết quả sorted theo score descending

Ưu điểm so với lexical search:
    - Hiểu được ngữ nghĩa ("hình phạt" ↔ "bị phạt tù")
    - Tìm được synonym, paraphrase
    - Hoạt động tốt với câu hỏi tự nhiên

Nhược điểm:
    - Không tốt với từ khóa chính xác (số điều luật, tên riêng)
    - Phụ thuộc vào chất lượng embedding model

Cài đặt:
    pip install sentence-transformers chromadb
"""

from pathlib import Path

# Import config từ Task 4 để đảm bảo consistency
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chromadb"
COLLECTION_NAME = "drug_law_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache model để không load lại mỗi lần gọi
_model = None
_collection = None


def _get_model():
    """Lazy-load embedding model (cache singleton)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _get_collection():
    """Lazy-load ChromaDB collection (cache singleton)."""
    global _collection
    if _collection is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(name=COLLECTION_NAME)
    return _collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector cosine similarity.

    Pipeline:
        query → embed(query) → ChromaDB.query(near_vector) → top_k results

    Args:
        query: Câu truy vấn (tiếng Việt hoặc tiếng Anh)
        top_k: Số lượng kết quả tối đa trả về

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score (0 → 1, cao = tốt)
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending (relevant nhất ở đầu).
    """
    # Bước 1: Embed query bằng cùng model ở Task 4
    model = _get_model()
    query_embedding = model.encode(query).tolist()

    # Bước 2: Query ChromaDB — tìm top_k vectors gần nhất
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    # Bước 3: Convert kết quả sang format chuẩn
    # ChromaDB trả về distance (thấp = gần = tốt)
    # Cosine distance = 1 - cosine_similarity → chuyển lại thành similarity
    output = []
    if results and results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Chuyển cosine distance → similarity score
            score = 1.0 - dist  # distance ∈ [0, 2] → score ∈ [-1, 1]
            score = max(0.0, score)  # Clamp về [0, 1]

            output.append({
                "content": doc,
                "score": round(score, 4),
                "metadata": meta,
            })

    # Đã sorted by distance ascending từ ChromaDB → score descending
    return output


if __name__ == "__main__":
    # Test với 3 câu hỏi mẫu
    test_queries = [
        "hình phạt cho tội tàng trữ ma tuý",
        "nghệ sĩ bị bắt vì sử dụng ma túy",
        "Điều 248 Bộ luật Hình sự",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print("-" * 60)
        results = semantic_search(q, top_k=5)
        for i, r in enumerate(results, 1):
            source = r["metadata"].get("source", "?")
            print(f"  {i}. [{r['score']:.3f}] ({source}) {r['content'][:80]}...")
        if not results:
            print("  (Không có kết quả — chạy Task 4 trước để index)")
