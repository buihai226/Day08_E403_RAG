"""
Task 8 — PageIndex Vectorless RAG.

PageIndex cho phép RAG mà KHÔNG cần vector store — sử dụng
structural understanding (cây mục lục + OCR) của document thay vì embedding.

Luồng dùng SDK chính thức (package `pageindex`, class `PageIndexClient`):
    1. submit_document(file_path)  → {'doc_id': ...}   (upload PDF)
    2. is_retrieval_ready(doc_id)  → bool              (đợi xử lý xong)
    3. submit_query(doc_id, query) → {'retrieval_id': ...}
    4. get_retrieval(retrieval_id) → status + results  (poll tới khi xong)

Đăng ký tại: https://pageindex.ai/
SDK & docs: https://github.com/VectifyAI/PageIndex

Cài đặt:
    pip install pageindex python-dotenv

Cấu hình:
    Thêm PAGEINDEX_API_KEY vào file .env
    Lưu ý: PageIndex nhận file PDF. Đặt các file .pdf cần index vào
    data/landing/legal/ (hoặc data/pageindex_docs/).
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
# Cache ánh xạ filename -> doc_id để khỏi upload lại mỗi lần
DOC_CACHE = Path(__file__).parent.parent / "data" / "pageindex_docs.json"

# Thời gian chờ tối đa (giây) cho upload xử lý xong / retrieval hoàn tất
READY_TIMEOUT = 120
POLL_INTERVAL = 3


# =============================================================================
# CLIENT
# =============================================================================

def _get_client():
    """Khởi tạo PageIndexClient. Trả None nếu thiếu key hoặc chưa cài SDK."""
    if not PAGEINDEX_API_KEY:
        print("  ⚠ PAGEINDEX_API_KEY chưa set trong .env")
        return None
    try:
        from pageindex import PageIndexClient
    except ImportError:
        print("  ⚠ pageindex chưa cài. Chạy: pip install pageindex")
        return None
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_doc_cache() -> dict:
    if DOC_CACHE.exists():
        try:
            return json.loads(DOC_CACHE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_doc_cache(cache: dict):
    DOC_CACHE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# =============================================================================
# UPLOAD DOCUMENTS LÊN PAGEINDEX
# =============================================================================

def upload_documents(force: bool = False) -> dict:
    """
    Upload các file PDF trong data/landing/legal/ lên PageIndex.

    PageIndex nhận PDF (submit_document). Mỗi file → 1 doc_id, được cache
    vào data/pageindex_docs.json để lần sau không upload lại.

    Args:
        force: True → upload lại kể cả đã có trong cache.

    Returns:
        dict {filename: doc_id}
    """
    client = _get_client()
    if client is None:
        return {}

    cache = {} if force else _load_doc_cache()

    pdf_files = sorted(LEGAL_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"  ⚠ Không tìm thấy file .pdf trong {LEGAL_DIR}")
        print("     PageIndex chỉ nhận PDF — hãy đặt file .pdf vào thư mục này.")
        return cache

    for pdf in pdf_files:
        if pdf.name in cache and not force:
            print(f"  ⏭ Đã có trong cache: {pdf.name} (doc_id={cache[pdf.name]})")
            continue
        try:
            print(f"  ⬆ Uploading: {pdf.name} ...")
            resp = client.submit_document(file_path=str(pdf))
            doc_id = resp.get("doc_id")
            cache[pdf.name] = doc_id
            print(f"     ✅ doc_id = {doc_id}")
        except Exception as e:
            print(f"     ❌ Upload failed ({pdf.name}): {e}")

    _save_doc_cache(cache)

    # Đợi các document xử lý xong (tree + OCR) để sẵn sàng retrieval
    for fname, doc_id in cache.items():
        if not doc_id:
            continue
        t0 = time.time()
        while time.time() - t0 < READY_TIMEOUT:
            try:
                if client.is_retrieval_ready(doc_id):
                    print(f"  ✅ Sẵn sàng: {fname}")
                    break
            except Exception as e:
                print(f"  ⚠ Lỗi kiểm tra trạng thái {fname}: {e}")
                break
            time.sleep(POLL_INTERVAL)
        else:
            print(f"  ⏳ Quá thời gian chờ xử lý: {fname}")

    print(f"\n  Tổng: {len([d for d in cache.values() if d])} document trên PageIndex")
    return cache


# =============================================================================
# PARSE KẾT QUẢ RETRIEVAL (phòng thủ — cấu trúc API có thể thay đổi)
# =============================================================================

def _extract_nodes(retrieval_result: dict) -> list[dict]:
    """
    Trích danh sách 'node' kết quả từ response get_retrieval, parse linh hoạt
    vì key có thể là 'results' / 'retrieval' / 'nodes' / 'sources'.
    """
    if not isinstance(retrieval_result, dict):
        return []
    for key in ("results", "retrieval", "retrieval_nodes", "nodes", "sources", "data"):
        val = retrieval_result.get(key)
        if isinstance(val, list) and val:
            return val
    return []


def _node_text(node: dict) -> str:
    for key in ("relevant_content", "content", "text", "node_text", "node_summary"):
        v = node.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _node_score(node: dict) -> float:
    for key in ("relevance_score", "score", "relevance"):
        v = node.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.5  # mặc định nếu API không trả score


# =============================================================================
# SEARCH (VECTORLESS RETRIEVAL)
# =============================================================================

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval qua PageIndex — dùng làm FALLBACK khi hybrid search
    không đủ tốt (Task 9).

    Với mỗi document đã upload (doc_id trong cache), submit_query rồi poll
    get_retrieval tới khi hoàn tất, gộp kết quả, trả top_k.

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': {'source': str},
            'source': 'pageindex'
        }
    """
    client = _get_client()
    if client is None:
        return []

    cache = _load_doc_cache()
    doc_ids = {f: d for f, d in cache.items() if d}
    if not doc_ids:
        print("  ⚠ Chưa có document nào trên PageIndex. Chạy upload_documents() trước.")
        return []

    results: list[dict] = []
    for fname, doc_id in doc_ids.items():
        try:
            sub = client.submit_query(doc_id=doc_id, query=query)
            retrieval_id = sub.get("retrieval_id")
            if not retrieval_id:
                continue

            # Poll tới khi retrieval hoàn tất
            t0 = time.time()
            retrieval = {}
            while time.time() - t0 < READY_TIMEOUT:
                retrieval = client.get_retrieval(retrieval_id)
                status = str(retrieval.get("status", "")).lower()
                if status in ("completed", "success", "done", "finished", ""):
                    if _extract_nodes(retrieval) or status:
                        break
                time.sleep(POLL_INTERVAL)

            for node in _extract_nodes(retrieval):
                text = _node_text(node)
                if not text:
                    continue
                results.append({
                    "content": text,
                    "score": round(_node_score(node), 4),
                    "metadata": {"source": fname},
                    "source": "pageindex",
                })
        except Exception as e:
            print(f"  ❌ PageIndex query failed ({fname}): {e}")

    # Sort theo score giảm dần, lấy top_k
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Task 8: PageIndex Vectorless RAG")
    print("=" * 60)

    if not PAGEINDEX_API_KEY:
        print()
        print("⚠ Chưa có PAGEINDEX_API_KEY!")
        print("Hướng dẫn:")
        print("  1. Đăng ký tại https://pageindex.ai/ và lấy API key")
        print("  2. Thêm vào .env:  PAGEINDEX_API_KEY=your_key")
        print("  3. Đặt file .pdf cần index vào data/landing/legal/")
        print("  4. Chạy lại script này")
    else:
        print("\n--- Upload Documents ---")
        upload_documents()

        print("\n--- Test Query ---")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] ({r['metadata']['source']}) "
                  f"{r['content'][:100]}...")
        if not results:
            print("  (Không có kết quả)")
