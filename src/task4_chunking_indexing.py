"""
Task 4 — Chunking & Indexing vào Vector Store (ChromaDB).

Lựa chọn & giải thích:

1. CHUNKING: RecursiveCharacterTextSplitter
   - Vì sao? Đây là splitter phổ biến nhất, xử lý tốt cả legal docs (có
     heading/điều khoản) và news articles (paragraphs). Nó thử tách theo
     ưu tiên: paragraph → newline → sentence → word, đảm bảo chunk có
     ngữ nghĩa tốt nhất có thể.
   - chunk_size=500: Đủ ngắn để embedding capture ngữ nghĩa cụ thể,
     đủ dài để chứa 1 ý hoàn chỉnh (1 điều luật ~ 200-600 ký tự).
   - chunk_overlap=50: Đảm bảo context liên tục giữa các chunk, tránh
     mất thông tin ở ranh giới.

2. EMBEDDING: sentence-transformers/all-MiniLM-L6-v2
   - Vì sao? Model nhẹ (80MB), chạy nhanh trên CPU, dimension 384 giúp
     index nhỏ gọn. Tuy hỗ trợ tiếng Việt kém hơn bge-m3, nhưng đủ
     tốt cho demo project. Nếu cần chất lượng cao hơn, đổi sang
     BAAI/bge-m3 (1024 dim, multilingual).

3. VECTOR STORE: ChromaDB (persistent, local)
   - Vì sao? Không cần Docker (khác Weaviate), cài pip là xong.
     Hỗ trợ persistent storage, metadata filtering, và cosine similarity
     search. Đủ cho RAG pipeline demo.

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import json
import hashlib
import pickle
from pathlib import Path
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
NEWS_LANDING_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chromadb"
CHUNKS_JSON = Path(__file__).parent.parent / "data" / "chunks.json"
# Cache embedding theo hash nội dung — tái sử dụng vector cho chunk không đổi
EMBED_CACHE = Path(__file__).parent.parent / "data" / "embeddings_cache.pkl"
# Chữ ký nguồn dữ liệu — để bỏ qua re-index khi data không thay đổi
INDEX_SIGNATURE = Path(__file__).parent.parent / "data" / ".index_signature"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn (xem docstring ở đầu file)
# =============================================================================

# Chunk size 500 ký tự: vừa đủ cho 1 điều luật hoặc 1 đoạn tin ngắn
CHUNK_SIZE = 500

# Overlap 50 ký tự: giữ context liên tục, ~10% chunk size
CHUNK_OVERLAP = 50

# RecursiveCharacterTextSplitter: tách theo thứ tự ưu tiên ngữ nghĩa
CHUNKING_METHOD = "recursive"

# all-MiniLM-L6-v2: nhẹ, nhanh, 384 dim — chạy được trên CPU
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ChromaDB: local, persistent, không cần Docker
VECTOR_STORE = "chromadb"
COLLECTION_NAME = "drug_law_docs"


# =============================================================================
# STEP 1: LOAD DOCUMENTS
# =============================================================================

def _load_news_from_landing() -> list[dict]:
    """
    Fallback: đọc thẳng bài báo đã crawl từ data/landing/news/*.json
    khi chưa có markdown news trong data/standardized/news/.

    Mỗi file JSON (tạo bởi Task 2) có dạng:
        {"url", "title", "content_markdown", ...}
    Ta ghép title + content thành 1 document để embedding.

    Returns:
        List of {'content': str, 'metadata': {'source', 'type': 'news'}}
    """
    docs = []
    if not NEWS_LANDING_DIR.exists():
        return docs

    for json_file in sorted(NEWS_LANDING_DIR.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            print(f"    ⚠ Bỏ qua JSON lỗi: {json_file.name}")
            continue

        title = data.get("title", "").strip()
        content = data.get("content_markdown", "").strip()
        if not content:
            continue

        # Ghép title vào đầu để giữ ngữ cảnh khi chunk/embedding
        full_text = f"# {title}\n\n{content}" if title else content

        docs.append({
            "content": full_text,
            "metadata": {
                "source": json_file.stem,   # vd: "article_01"
                "type": "news",
                "url": data.get("url", ""),
            }
        })

    return docs


def load_documents() -> list[dict]:
    """
    Đọc documents để index, bao gồm CẢ văn bản pháp luật VÀ bài báo nghệ sĩ.

    Nguồn:
        1. data/standardized/**/*.md  — markdown đã chuẩn hoá (Task 3)
        2. Fallback: data/landing/news/*.json — nếu chưa có markdown news,
           đọc thẳng bài báo crawl được từ Task 2 để vẫn embedding được news.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []

    if STANDARDIZED_DIR.exists():
        for md_file in STANDARDIZED_DIR.rglob("*.md"):
            content = md_file.read_text(encoding="utf-8")

            # Xác định loại document từ thư mục cha
            doc_type = "legal" if "legal" in str(md_file) else "news"

            documents.append({
                "content": content,
                "metadata": {
                    "source": md_file.name,    # vd: "luat-phong-chong-ma-tuy-2025.md"
                    "type": doc_type,           # "legal" hoặc "news"
                }
            })
    else:
        print("  ⚠ Thư mục data/standardized/ chưa có. Chạy Task 3 trước.")

    # Fallback: nếu chưa có bài báo nào trong standardized → đọc thẳng JSON crawl
    has_news = any(d["metadata"]["type"] == "news" for d in documents)
    if not has_news:
        news_docs = _load_news_from_landing()
        if news_docs:
            print(f"  ℹ Chưa có markdown news → đọc thẳng {len(news_docs)} "
                  f"bài báo từ data/landing/news/")
            documents.extend(news_docs)

    return documents


# =============================================================================
# STEP 2: CHUNKING
# =============================================================================

def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Separators theo thứ tự ưu tiên:
        1. "\\n\\n" — tách theo paragraph (tốt nhất)
        2. "\\n"   — tách theo dòng
        3. ". "    — tách theo câu
        4. " "     — tách theo từ
        5. ""      — tách theo ký tự (fallback cuối cùng)

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Ưu tiên tách theo paragraph
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            # Bỏ chunk quá ngắn (< 20 ký tự) — thường là noise
            if len(chunk_text.strip()) < 20:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,  # Thứ tự chunk trong document gốc
                }
            })

    return chunks


# =============================================================================
# STEP 3: EMBEDDING
# =============================================================================

def _content_hash(text: str) -> str:
    """Hash nội dung + tên model làm khóa cache (model đổi → vector phải khác)."""
    key = f"{EMBEDDING_MODEL}::{text}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _load_embed_cache() -> dict:
    """Đọc cache embedding từ disk. Trả dict rỗng nếu chưa có/hỏng."""
    if not EMBED_CACHE.exists():
        return {}
    try:
        with open(EMBED_CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:
        print("  ⚠ Cache embedding hỏng — sẽ tạo lại từ đầu.")
        return {}


def _save_embed_cache(cache: dict):
    """Lưu cache embedding ra disk."""
    EMBED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBED_CACHE, "wb") as f:
        pickle.dump(cache, f)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed chunks bằng sentence-transformers, CÓ CACHE theo hash nội dung.

    Cơ chế:
        - Mỗi chunk được hash (md5 của model+nội dung).
        - Chunk nào đã có trong cache → lấy lại vector, KHÔNG encode lại.
        - Chỉ encode những chunk mới/đổi nội dung.
        - Nếu TẤT CẢ đã có cache → không cần load model (tiết kiệm thời gian).

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    cache = _load_embed_cache()

    # Tính hash + tách ra phần đã cache và phần cần encode
    hashes = [_content_hash(c["content"]) for c in chunks]
    missing_idx = [i for i, h in enumerate(hashes) if h not in cache]

    print(f"  Cache: {len(chunks) - len(missing_idx)}/{len(chunks)} chunks dùng lại, "
          f"{len(missing_idx)} cần encode mới")

    # Chỉ load model + encode khi có chunk mới
    if missing_idx:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading model: {EMBEDDING_MODEL}...")
        model = SentenceTransformer(EMBEDDING_MODEL)

        texts_to_encode = [chunks[i]["content"] for i in missing_idx]
        print(f"  Encoding {len(texts_to_encode)} chunks...")
        new_embeddings = model.encode(
            texts_to_encode,
            show_progress_bar=True,
            batch_size=32,
        )

        # Cập nhật cache
        for idx, emb in zip(missing_idx, new_embeddings):
            cache[hashes[idx]] = emb.tolist()
        _save_embed_cache(cache)

    # Gán embedding cho từng chunk (từ cache)
    for chunk, h in zip(chunks, hashes):
        chunk["embedding"] = cache[h]

    return chunks


# =============================================================================
# STEP 4: INDEX VÀO CHROMADB
# =============================================================================

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào ChromaDB (persistent storage).

    ChromaDB tự động tạo index cho cosine similarity search.
    Mỗi chunk được lưu với:
        - id: unique identifier
        - embedding: vector 384 chiều
        - document: nội dung text
        - metadata: source, type, chunk_index
    """
    import chromadb

    # Tạo/mở persistent ChromaDB client
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Xóa collection cũ nếu tồn tại (để re-index clean)
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"  🗑 Xóa collection cũ: {COLLECTION_NAME}")
    except Exception:
        pass

    # Tạo collection mới với cosine distance (phổ biến nhất cho NLP)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Cosine similarity
    )

    # Chuẩn bị data cho batch insert
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Batch insert (ChromaDB tự xử lý batching)
    # ChromaDB giới hạn batch size ~41666, chia nhỏ nếu cần
    batch_size = 5000
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    print(f"  ✅ Indexed {collection.count()} chunks vào ChromaDB")
    print(f"     Collection: {COLLECTION_NAME}")
    print(f"     Storage: {CHROMA_DIR}")

    # Đóng client
    del client


def save_chunks_json(chunks: list[dict]):
    """
    Lưu chunks ra file JSON (dùng cho BM25 ở Task 6).
    Không lưu embedding vào JSON (quá lớn) — chỉ lưu content + metadata.
    """
    chunks_no_emb = [
        {"content": c["content"], "metadata": c["metadata"]}
        for c in chunks
    ]
    CHUNKS_JSON.write_text(
        json.dumps(chunks_no_emb, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  💾 Saved chunks JSON: {CHUNKS_JSON} ({len(chunks_no_emb)} chunks)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def _compute_signature(docs: list[dict]) -> str:
    """
    Tính 'chữ ký' của bộ tài liệu nguồn + cấu hình chunk.
    Nếu chữ ký không đổi giữa 2 lần chạy → data không thay đổi → có thể bỏ qua.
    """
    h = hashlib.md5()
    h.update(f"{CHUNK_SIZE}|{CHUNK_OVERLAP}|{EMBEDDING_MODEL}".encode())
    for d in sorted(docs, key=lambda x: x["metadata"]["source"]):
        h.update(d["metadata"]["source"].encode("utf-8"))
        h.update(_content_hash(d["content"]).encode("utf-8"))
    return h.hexdigest()


def _collection_has_data() -> bool:
    """Kiểm tra ChromaDB collection đã tồn tại và có dữ liệu chưa."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        col = client.get_collection(name=COLLECTION_NAME)
        return col.count() > 0
    except Exception:
        return False


def run_pipeline(force: bool = False):
    """
    Chạy pipeline: load → chunk → embed → index.

    Args:
        force: True → luôn index lại, kể cả khi data không đổi.
               False → bỏ qua nếu data + config không thay đổi (nhanh).
    """
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE} → {CHROMA_DIR}")
    print("=" * 60)

    # Step 1: Load documents
    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")
    for d in docs:
        print(f"    {d['metadata']['type']}/{d['metadata']['source']} "
              f"({len(d['content']):,} chars)")

    if not docs:
        print("⚠ Không có documents. Chạy Task 3 trước!")
        return

    # ---- Kiểm tra có cần index lại không ----
    signature = _compute_signature(docs)
    old_signature = (
        INDEX_SIGNATURE.read_text(encoding="utf-8").strip()
        if INDEX_SIGNATURE.exists() else ""
    )
    if (not force and signature == old_signature
            and _collection_has_data() and CHUNKS_JSON.exists()):
        print("\n✅ Data + config không đổi → BỎ QUA re-index (đã có sẵn trong ChromaDB).")
        print("   Muốn ép index lại: python src\\task4_chunking_indexing.py --force")
        return

    # Step 2: Chunk
    chunks = chunk_documents(docs)
    print(f"\n✓ Created {len(chunks)} chunks")

    # Step 3: Embed (có cache)
    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks (dim={EMBEDDING_DIM})")

    # Step 4: Index
    index_to_vectorstore(chunks)

    # Bonus: lưu chunks JSON cho BM25 (Task 6)
    save_chunks_json(chunks)

    # Lưu chữ ký để lần sau biết data có đổi không
    INDEX_SIGNATURE.write_text(signature, encoding="utf-8")

    print("\n✅ Pipeline hoàn tất!")


if __name__ == "__main__":
    import sys
    force_flag = "--force" in sys.argv
    run_pipeline(force=force_flag)
