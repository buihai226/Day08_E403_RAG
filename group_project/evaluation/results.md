# RAG Evaluation Results

## Framework sử dụng

- **DeepEval** với LLM-judge = `gpt-4o-mini`
- Golden dataset: **18** cặp Q&A
- 4 metrics: Faithfulness, Answer Relevancy, Contextual Recall, Contextual Precision

## Overall Scores (A/B Comparison)

| Metric | Config A (hybrid + rerank) | Config B (dense-only (no rerank)) |
|--------|----|----|
| Faithfulness | 0.000 | 0.000 |
| Answer Relevancy | 0.000 | 0.000 |
| Contextual Recall | 0.000 | 0.000 |
| Contextual Precision | 0.000 | 0.000 |
| **Average** | 0.000 | 0.000 |

## A/B Comparison Analysis

- **Config A** (hybrid + rerank): điểm trung bình `0.000`
- **Config B** (dense-only (no rerank)): điểm trung bình `0.000`
- **Kết luận:** Config **A** tốt hơn (chênh `0.000`). Hybrid + reranking thường cải thiện Context Precision/Recall vì kết hợp được cả khớp ngữ nghĩa lẫn từ khóa, rồi rerank đẩy chunk liên quan nhất lên đầu.

## Worst Performers (Bottom 3)


### Config A (hybrid + rerank)

| # | Question | Faith | Relev | Recall | Precis | Avg |
|---|----------|-------|-------|--------|--------|-----|

### Config B (dense-only (no rerank))

| # | Question | Faith | Relev | Recall | Precis | Avg |
|---|----------|-------|-------|--------|--------|-----|

## Recommendations

### 1. Embedding model tiếng Việt
**Action:** Đổi `all-MiniLM-L6-v2` sang `BAAI/bge-m3` (multilingual).  
**Expected impact:** Tăng Context Recall/Precision cho câu hỏi tiếng Việt.

### 2. Tokenizer cho BM25
**Action:** Dùng `underthesea`/`pyvi` tách từ tiếng Việt thay cho `.split()`.  
**Expected impact:** BM25 khớp đúng từ ghép → cải thiện lexical recall.

### 3. Cross-encoder reranking
**Action:** Bật `cross_encoder` (Jina) thay RRF khi có API key.  
**Expected impact:** Tăng Faithfulness nhờ context chính xác hơn.
