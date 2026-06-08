# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
                        ┌─────────────────────────┐
                        │   Người dùng (Streamlit) │
                        │        app.py            │
                        └────────────┬─────────────┘
                                     │ câu hỏi
                                     ▼
   ┌──────────────────────── RETRIEVAL PIPELINE (Task 9) ─────────────────────────┐
   │                                                                              │
   │   ┌────────────────────┐        ┌────────────────────┐                       │
   │   │ Semantic Search     │        │ Lexical Search      │                      │
   │   │ (Task 5, ChromaDB)  │        │ (Task 6, BM25)      │                      │
   │   └─────────┬──────────┘        └─────────┬──────────┘                       │
   │             └────────────┬─────────────────┘                                 │
   │                          ▼                                                    │
   │                 ┌──────────────────┐      ┌─────────────────────┐            │
   │                 │ Merge — RRF       │ ───▶ │ Rerank (Task 7)      │           │
   │                 │ (Task 7)          │      │ RRF / MMR / X-encoder│           │
   │                 └──────────────────┘      └─────────┬───────────┘            │
   │                                                      │  score < ngưỡng?       │
   │                                                      ▼                        │
   │                                          ┌────────────────────────┐          │
   │                                          │ Fallback: PageIndex     │          │
   │                                          │ Vectorless (Task 8)     │          │
   │                                          └────────────────────────┘          │
   └──────────────────────────────────┬───────────────────────────────────────────┘
                                       │ top-k chunks
                                       ▼
                        ┌─────────────────────────────┐
                        │ Generation (Task 10)         │
                        │ reorder (lost-in-the-middle) │
                        │ → prompt → LLM → citation     │
                        └────────────┬────────────────┘
                                     ▼
                        ┌─────────────────────────────┐
                        │ Câu trả lời + nguồn (UI)     │
                        └─────────────────────────────┘

   Dữ liệu: data/landing → data/standardized (Task 3) → ChromaDB + chunks.json (Task 4)
   Đánh giá: group_project/evaluation/eval_pipeline.py (DeepEval, 4 metrics, A/B)
```

**Luồng dữ liệu (offline / index):**
`Task 1` thu thập luật (PDF/DOCX/TXT) + `Task 2` crawl báo (JSON) → `Task 3` convert Markdown →
`Task 4` chunk + embed (MiniLM) → ChromaDB (`data/chromadb/`) + `chunks.json` (cho BM25).

**Luồng truy vấn (online):** xem diagram phía trên — hybrid retrieval → rerank → fallback →
generation có citation, demo trực quan từng bước trong `app.py`.

---

## Phân Công Công Việc

> Dự án thực hiện cá nhân (một người làm toàn bộ pipeline + chatbot + evaluation).

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| (điền tên) | (điền MSSV) | Task 1–4: thu thập, chuẩn hóa, chunking & indexing | ✅ Done |
| (điền tên) | (điền MSSV) | Task 5–7: semantic / lexical search + reranking | ✅ Done |
| (điền tên) | (điền MSSV) | Task 8–10: PageIndex, retrieval pipeline, generation | ✅ Done |
| (điền tên) | (điền MSSV) | Chatbot UI (`app.py`) + Evaluation pipeline (DeepEval) | ✅ Done |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
