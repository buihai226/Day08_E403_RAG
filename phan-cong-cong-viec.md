# Phân Công Công Việc Nhóm — RAG Pipeline (Pháp luật & tin tức ma túy)

## Tổng quan kiến trúc 

```
Thu thập (Task 1-2) → Convert Markdown (Task 3) → Chunk + Embed → ChromaDB (Task 4)
        │
Truy vấn: Semantic (Task 5) + BM25 (Task 6) → Merge RRF + Rerank (Task 7)
        → Fallback PageIndex (Task 8) → Pipeline hợp nhất (Task 9)
        → Generation có citation (Task 10) → Chatbot UI (app.py)
        → Đánh giá DeepEval 4 metrics + A/B (group_project/evaluation)
```

---

## Bảng phân công

| Thành viên | MSSV | Vai trò chính |
|-----------|------|----------------|
| Bùi Xuân Hải | 2A202600862 | Kiến trúc tổng thể, pipeline lõi, tích hợp, chatbot |
| Nguyễn Như Yến Phương | 2A202600616 | Thu thập & chuẩn hóa dữ liệu (Task 1–3) |
| Lê Sỹ Minh Quang | 2A202600931 | Indexing & semantic search (Task 4–5) |
| Nguyễn Quang Khánh An | 2A202600698 | Lexical, reranking & vectorless (Task 6–8) |
| Đào Duy Quyền | 2A202600676 | Evaluation, kiểm thử & trình bày |

---

## Bùi Xuân Hải — Kiến trúc & Pipeline lõi & Tích hợp

**MSSV:** 2A202600862

**Nhiệm vụ:**
- Thiết kế kiến trúc tổng thể của hệ thống RAG, quy ước format dữ liệu chung
  (`{'content', 'score', 'metadata'}`) để các module ghép nối được với nhau.
- **Task 9 — Retrieval pipeline hợp nhất:** hybrid (semantic + lexical) → merge RRF
  → rerank → logic fallback theo ngưỡng (`src/task9_retrieval_pipeline.py`).
- **Task 10 — Generation có citation:** reorder chống "lost in the middle",
  prompt có citation, hỗ trợ hội thoại nhiều lượt (`src/task10_generation.py`).
- **Chatbot UI (`app.py`):** giao diện Streamlit, hiển thị từng bước pipeline + nguồn.
- **Tích hợp** code của các thành viên vào pipeline thống nhất, review & merge PR.
- Viết phần kiến trúc trong README chung.

**File phụ trách:** `src/task9_*.py`, `src/task10_*.py`, `app.py`, `README` (kiến trúc).

**Tiêu chí hoàn thành:** pipeline chạy end-to-end, chatbot demo được, nhóm tích hợp xong.

---

## Nguyễn Như Yến Phương — Thu thập & Chuẩn hóa dữ liệu

**MSSV:** 2A202600616

**Nhiệm vụ:**
- **Task 1:** thu thập ≥3 văn bản pháp luật (PDF/DOCX) về ma túy, đặt tên rõ ràng.
- **Task 2:** crawl ≥5 bài báo về nghệ sĩ liên quan ma túy, lưu JSON kèm metadata.
- **Task 3:** convert toàn bộ `landing/` → Markdown (`standardized/`), xử lý cả
  PDF/DOCX (MarkItDown) lẫn JSON; bảo đảm dedup theo tên file.
- Kiểm tra chất lượng nội dung crawl (chống dính trang chủ/nội dung rác).

**File phụ trách:** `src/task1_*.py`, `src/task2_*.py`, `src/task3_*.py`,
`src/make_legal_pdfs.py`.

**Tiêu chí hoàn thành:** `test_task1`, `test_task2`, `test_task3` pass; data sạch.

---

## Lê Sỹ Minh Quang — Indexing & Semantic Search

**MSSV:** 2A202600931

**Nhiệm vụ:**
- **Task 4:** chunking (RecursiveCharacterTextSplitter), embedding
  (`all-MiniLM-L6-v2`), index vào ChromaDB; giải thích lựa chọn chunk_size/overlap.
- Tối ưu: cache embedding theo hash + cơ chế bỏ qua re-index khi data không đổi.
- **Task 5:** module semantic search (dense retrieval) trên ChromaDB.
- Khảo sát đổi sang embedding tiếng Việt (`bge-m3`) và so sánh.
- **Bonus — HyDE** (Hypothetical Document Embeddings): `src/hyde.py`.

**File phụ trách:** `src/task4_*.py`, `src/task5_*.py`, `src/hyde.py`.

**Tiêu chí hoàn thành:** `test_task4`, `test_task5` pass; index đủ legal + news.

---

## Nguyễn Quang Khánh An — Lexical, Reranking & Vectorless

**MSSV:** 2A202600698

**Nhiệm vụ:**
- **Task 6:** lexical search BM25; **bonus**: thêm TF-IDF và giải thích khác biệt
  BM25 vs TF-IDF trong buổi demo.
- **Task 7:** reranking — RRF + MMR + cross-encoder (Jina).
- **Task 8:** PageIndex vectorless (upload PDF, query, fallback); cấu hình API key.

**File phụ trách:** `src/task6_*.py`, `src/task7_*.py`, `src/task8_*.py`.

**Tiêu chí hoàn thành:** `test_task6`, `test_task7`, `test_task8` pass; trình bày
được cơ chế BM25 vs TF-IDF.

---

## Đào Duy Quyền — Evaluation, Kiểm thử & Trình bày

**MSSV:** 2A202600676

**Nhiệm vụ:**
- **Evaluation pipeline (DeepEval):** hoàn thiện golden dataset ≥15 cặp Q&A,
  chạy 4 metrics (Faithfulness, Answer Relevancy, Context Recall/Precision).
- **So sánh A/B** ≥2 config (hybrid+rerank vs dense-only) và viết `results.md`
  có phân tích worst performers + đề xuất cải tiến.
- Chạy & duy trì bộ test (`pytest tests/`), bảo đảm 35/35 pass trước khi nộp.
- Chuẩn bị slide + kịch bản demo cho buổi trình bày; viết phần README chung
  (cài đặt, hướng dẫn chạy, phân công).

**File phụ trách:** `group_project/evaluation/*`, `tests/test_individual.py`,
slide demo.

**Tiêu chí hoàn thành:** `results.md` có số liệu thật + phân tích; demo trơn tru.

---

## Bonus (chia thêm, ai xong việc chính thì nhận)

| Hạng mục bonus | Người làm |
|----------------|-----------|
| Lexical khác BM25 (TF-IDF) + giải thích | Nguyễn Quang Khánh An |
| HyDE (Hypothetical Document Embeddings) | Lê Sỹ Minh Quang |
| Conversation memory (multi-turn) | Bùi Xuân Hải |
| UI/UX: hiển thị source, score, highlight | Bùi Xuân Hải |
| Deploy online (HF Spaces / Streamlit Cloud) | Đào Duy Quyền |

---

## Quy trình làm việc chung

1. Mỗi người làm trên nhánh `personal/<MSSV>` rồi tạo Pull Request.
2. Bùi Xuân Hải review và tích hợp vào nhánh chính.
3. Trước khi nộp: chạy `pytest tests/ -v` (đủ 35 pass) và chạy full evaluation.
4. Cập nhật bảng phân công (tên + MSSV) và README chung.

## Checklist nộp bài

- [x] 35/35 test cá nhân pass
- [x] Chatbot `app.py` demo được
- [x] `results.md` có số liệu eval + A/B + phân tích
- [x] README chung có kiến trúc + phân công
- [x] (Bonus) HyDE / TF-IDF / conversation memory
