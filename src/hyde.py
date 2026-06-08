"""
HyDE — Hypothetical Document Embeddings (BONUS).

Ý tưởng (Gao et al., 2022, "Precise Zero-Shot Dense Retrieval without
Relevance Labels"):
    - Câu hỏi thường NGẮN, ít từ khóa → embedding của nó không sát với
      "tài liệu đích" trong vector space.
    - Thay vì embed câu hỏi, ta cho LLM SINH một câu trả lời/đoạn văn GIẢ ĐỊNH
      (hypothetical document) cho câu hỏi đó, rồi embed đoạn văn này để search.
    - Đoạn văn giả định giống tài liệu thật hơn → semantic search khớp tốt hơn,
      đặc biệt khi corpus và câu hỏi cùng ngôn ngữ.

Lưu ý: HyDE cần LLM để sinh hypothetical doc (OPENAI_API_KEY / GOOGLE_API_KEY).
Nếu không có key → fallback dùng chính câu hỏi gốc để search.

Cách dùng:
    from src.hyde import hyde_search
    results = hyde_search("hình phạt tàng trữ ma túy", top_k=5)
"""

import sys
from pathlib import Path

_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.task5_semantic_search import semantic_search
from src.task10_generation import _call_llm


HYDE_SYSTEM_PROMPT = (
    "Bạn là chuyên gia pháp luật Việt Nam về ma túy. "
    "Hãy viết một đoạn văn ngắn (3-5 câu) trả lời GIẢ ĐỊNH cho câu hỏi, "
    "viết như trích từ văn bản luật hoặc bài báo, dùng thuật ngữ chuyên ngành. "
    "KHÔNG cần chính xác tuyệt đối — mục đích là tạo đoạn văn giống tài liệu thật "
    "để phục vụ tìm kiếm. Trả lời bằng tiếng Việt."
)


def generate_hypothetical_document(query: str) -> str:
    """
    Sinh 'tài liệu giả định' cho câu hỏi bằng LLM.
    Nếu LLM lỗi/không có key → trả về chính query (degrade an toàn).
    """
    try:
        doc = _call_llm(HYDE_SYSTEM_PROMPT, f"Câu hỏi: {query}")
        if doc and "Không thể gọi LLM" not in doc and len(doc.strip()) > 20:
            return doc.strip()
    except Exception as e:
        print(f"  ⚠ HyDE generate lỗi: {e}")
    # Fallback: dùng query gốc
    return query


def hyde_search(query: str, top_k: int = 10, verbose: bool = False) -> list[dict]:
    """
    Semantic search theo kiểu HyDE.

    1. LLM sinh hypothetical document từ query.
    2. Embed hypothetical document đó và semantic search trên ChromaDB.

    Args:
        query: Câu truy vấn gốc
        top_k: Số kết quả
        verbose: In hypothetical document ra để quan sát

    Returns:
        List of {'content', 'score', 'metadata'} — như semantic_search.
    """
    hypo_doc = generate_hypothetical_document(query)
    if verbose:
        print(f"  🧪 Hypothetical document:\n  {hypo_doc[:300]}\n")

    # Embed hypo_doc (semantic_search nhận text bất kỳ làm "query")
    return semantic_search(hypo_doc, top_k=top_k)


if __name__ == "__main__":
    tests = [
        "hình phạt cho tội tàng trữ trái phép chất ma túy",
        "chất ma túy được định nghĩa thế nào",
    ]
    for q in tests:
        print(f"\n{'='*60}\nQuery: {q}\n{'-'*60}")
        results = hyde_search(q, top_k=3, verbose=True)
        for i, r in enumerate(results, 1):
            src = r["metadata"].get("source", "?")
            print(f"  {i}. [{r['score']:.3f}] ({src}) {r['content'][:70]}...")
        if not results:
            print("  (Không có kết quả — chạy Task 4 trước)")
