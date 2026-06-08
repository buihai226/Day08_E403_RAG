"""
RAG Evaluation Pipeline — DeepEval.

Đánh giá chất lượng RAG pipeline của dự án bằng DeepEval với 4 metrics:
    - Faithfulness        : câu trả lời có bám đúng context không?
    - Answer Relevancy    : câu trả lời có đúng trọng tâm câu hỏi không?
    - Contextual Recall   : retriever có lấy đủ evidence (so với expected) không?
    - Contextual Precision: context lấy về có sắp xếp đúng/hữu ích không?

So sánh A/B 2 cấu hình retrieval:
    - Config A: hybrid (semantic + BM25) + reranking  (pipeline Task 9)
    - Config B: dense-only (chỉ semantic search, không rerank)

Kết quả xuất ra results.md: bảng điểm, so sánh A/B, worst performers, đề xuất.

Yêu cầu:
    pip install deepeval        (đã có trong requirements.txt)
    OPENAI_API_KEY trong .env   (DeepEval dùng làm LLM-judge, mặc định gpt-4o-mini)

Chạy:
    # Chạy đầy đủ (2 config, toàn bộ golden dataset)
    python group_project/evaluation/eval_pipeline.py

    # Chạy nhanh với 5 câu đầu để test
    python group_project/evaluation/eval_pipeline.py --limit 5

    # Chỉ chạy 1 config
    python group_project/evaluation/eval_pipeline.py --config A
"""

import os
import sys
import json
import argparse
from pathlib import Path
from statistics import mean

from dotenv import load_dotenv

load_dotenv()

# Tắt telemetry của DeepEval cho gọn log
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_RESULTS_FOLDER", str(Path(__file__).parent))

# Cho phép import package src/
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Model dùng làm LLM-judge cho các metric (rẻ, đủ tốt)
JUDGE_MODEL = "gpt-4o-mini"

# Cấu hình A/B
CONFIGS = {
    "A": {
        "name": "hybrid + rerank",
        "mode": "hybrid",
        "use_reranking": True,
    },
    "B": {
        "name": "dense-only (no rerank)",
        "mode": "dense",
        "use_reranking": False,
    },
}

METRIC_NAMES = ["faithfulness", "answer_relevancy", "contextual_recall",
                "contextual_precision"]


# =============================================================================
# LOAD DATA
# =============================================================================

def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# RAG RUNNER — chạy pipeline theo từng config
# =============================================================================

def rag_answer(query: str, config: dict, top_k: int = 5) -> tuple[str, list[str]]:
    """
    Sinh câu trả lời + context theo cấu hình retrieval.

    Returns:
        (answer, contexts) — contexts là list nội dung chunk đã dùng.
    """
    from src.task5_semantic_search import semantic_search
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import (
        reorder_for_llm, format_context, _call_llm, SYSTEM_PROMPT,
    )

    if config["mode"] == "hybrid":
        chunks = retrieve(query, top_k=top_k, use_reranking=config["use_reranking"])
    else:  # dense-only
        chunks = semantic_search(query, top_k=top_k)
        for c in chunks:
            c["source"] = "dense"

    if not chunks:
        return "Tôi không tìm thấy thông tin liên quan.", []

    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"
    answer = _call_llm(SYSTEM_PROMPT, user_message)

    contexts = [c["content"] for c in chunks]
    return answer, contexts


# =============================================================================
# DEEPEVAL METRICS
# =============================================================================

def _build_metrics():
    """Khởi tạo 4 metric DeepEval (LLM-judge = gpt-4o-mini)."""
    from deepeval.metrics import (
        FaithfulnessMetric,
        AnswerRelevancyMetric,
        ContextualRecallMetric,
        ContextualPrecisionMetric,
    )
    return {
        "faithfulness": FaithfulnessMetric(threshold=0.7, model=JUDGE_MODEL),
        "answer_relevancy": AnswerRelevancyMetric(threshold=0.7, model=JUDGE_MODEL),
        "contextual_recall": ContextualRecallMetric(threshold=0.7, model=JUDGE_MODEL),
        "contextual_precision": ContextualPrecisionMetric(threshold=0.7, model=JUDGE_MODEL),
    }


def evaluate_config(config_key: str, dataset: list[dict], limit: int = 0) -> dict:
    """
    Chạy RAG theo 1 config trên golden dataset và chấm 4 metric mỗi câu.

    Returns:
        {
          'config': config_key,
          'per_question': [{'question','scores':{metric:score}, 'answer'}],
          'averages': {metric: avg_score}
        }
    """
    from deepeval.test_case import LLMTestCase

    config = CONFIGS[config_key]
    items = dataset[:limit] if limit else dataset

    print(f"\n{'='*70}")
    print(f"  CONFIG {config_key}: {config['name']} — {len(items)} câu hỏi")
    print(f"{'='*70}")

    metrics = _build_metrics()
    per_question = []

    for i, item in enumerate(items, 1):
        q = item["question"]
        print(f"\n  [{i}/{len(items)}] {q[:60]}...")

        try:
            answer, contexts = rag_answer(q, config)
        except Exception as e:
            print(f"    ❌ RAG lỗi: {e}")
            continue

        test_case = LLMTestCase(
            input=q,
            actual_output=answer,
            expected_output=item["expected_answer"],
            retrieval_context=contexts,
        )

        scores = {}
        for mname, metric in metrics.items():
            try:
                metric.measure(test_case)
                scores[mname] = float(metric.score) if metric.score is not None else 0.0
            except Exception as e:
                print(f"    ⚠ metric {mname} lỗi: {e}")
                scores[mname] = 0.0
        avg = mean(scores.values()) if scores else 0.0
        print(f"    scores: " + ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
              + f" | avg={avg:.2f}")

        per_question.append({
            "question": q,
            "answer": answer,
            "scores": scores,
            "avg": avg,
        })

    # Tính trung bình mỗi metric
    averages = {}
    for m in METRIC_NAMES:
        vals = [pq["scores"].get(m, 0.0) for pq in per_question if pq["scores"]]
        averages[m] = mean(vals) if vals else 0.0
    averages["overall"] = mean(averages.values()) if averages else 0.0

    return {"config": config_key, "per_question": per_question, "averages": averages}


# =============================================================================
# EXPORT RESULTS
# =============================================================================

def export_results(results: dict[str, dict]):
    """Ghi results.md: bảng điểm, so sánh A/B, worst performers, đề xuất."""
    lines = []
    lines.append("# RAG Evaluation Results\n")
    lines.append("## Framework sử dụng\n")
    lines.append(f"- **DeepEval** với LLM-judge = `{JUDGE_MODEL}`")
    lines.append(f"- Golden dataset: **{len(load_golden_dataset())}** cặp Q&A")
    lines.append(f"- 4 metrics: Faithfulness, Answer Relevancy, Contextual Recall, "
                 f"Contextual Precision\n")

    # --- Bảng tổng điểm ---
    lines.append("## Overall Scores (A/B Comparison)\n")
    keys = list(results.keys())
    header = "| Metric | " + " | ".join(
        f"Config {k} ({CONFIGS[k]['name']})" for k in keys) + " |"
    sep = "|--------|" + "|".join(["----"] * len(keys)) + "|"
    lines.append(header)
    lines.append(sep)
    metric_labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "contextual_recall": "Contextual Recall",
        "contextual_precision": "Contextual Precision",
        "overall": "**Average**",
    }
    for m, label in metric_labels.items():
        row = f"| {label} | " + " | ".join(
            f"{results[k]['averages'].get(m, 0):.3f}" for k in keys) + " |"
        lines.append(row)

    # --- Phân tích A/B ---
    lines.append("\n## A/B Comparison Analysis\n")
    if len(keys) >= 2:
        a, b = keys[0], keys[1]
        oa = results[a]["averages"]["overall"]
        ob = results[b]["averages"]["overall"]
        better = a if oa >= ob else b
        lines.append(f"- **Config {a}** ({CONFIGS[a]['name']}): điểm trung bình "
                     f"`{oa:.3f}`")
        lines.append(f"- **Config {b}** ({CONFIGS[b]['name']}): điểm trung bình "
                     f"`{ob:.3f}`")
        lines.append(f"- **Kết luận:** Config **{better}** tốt hơn "
                     f"(chênh `{abs(oa-ob):.3f}`). "
                     f"Hybrid + reranking thường cải thiện Context Precision/Recall "
                     f"vì kết hợp được cả khớp ngữ nghĩa lẫn từ khóa, rồi rerank "
                     f"đẩy chunk liên quan nhất lên đầu.\n")
    else:
        lines.append("(Chỉ chạy 1 config — không có so sánh A/B.)\n")

    # --- Worst performers ---
    lines.append("## Worst Performers (Bottom 3)\n")
    for k in keys:
        lines.append(f"\n### Config {k} ({CONFIGS[k]['name']})\n")
        pqs = sorted(results[k]["per_question"], key=lambda x: x["avg"])[:3]
        lines.append("| # | Question | Faith | Relev | Recall | Precis | Avg |")
        lines.append("|---|----------|-------|-------|--------|--------|-----|")
        for i, pq in enumerate(pqs, 1):
            s = pq["scores"]
            lines.append(
                f"| {i} | {pq['question'][:50]} | "
                f"{s.get('faithfulness',0):.2f} | {s.get('answer_relevancy',0):.2f} | "
                f"{s.get('contextual_recall',0):.2f} | {s.get('contextual_precision',0):.2f} | "
                f"{pq['avg']:.2f} |"
            )

    # --- Đề xuất ---
    lines.append("\n## Recommendations\n")
    lines.append("### 1. Embedding model tiếng Việt")
    lines.append("**Action:** Đổi `all-MiniLM-L6-v2` sang `BAAI/bge-m3` (multilingual).  ")
    lines.append("**Expected impact:** Tăng Context Recall/Precision cho câu hỏi tiếng Việt.\n")
    lines.append("### 2. Tokenizer cho BM25")
    lines.append("**Action:** Dùng `underthesea`/`pyvi` tách từ tiếng Việt thay cho `.split()`.  ")
    lines.append("**Expected impact:** BM25 khớp đúng từ ghép → cải thiện lexical recall.\n")
    lines.append("### 3. Cross-encoder reranking")
    lines.append("**Action:** Bật `cross_encoder` (Jina) thay RRF khi có API key.  ")
    lines.append("**Expected impact:** Tăng Faithfulness nhờ context chính xác hơn.\n")

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Đã ghi báo cáo: {RESULTS_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="RAG Evaluation với DeepEval")
    parser.add_argument("--limit", type=int, default=0,
                        help="Giới hạn số câu hỏi (0 = tất cả)")
    parser.add_argument("--config", choices=["A", "B", "both"], default="both",
                        help="Chạy config nào (mặc định: both)")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ Chưa có OPENAI_API_KEY trong .env — DeepEval cần để làm LLM-judge.")
        print("  Thêm OPENAI_API_KEY=sk-... vào .env rồi chạy lại.")
        return

    try:
        import deepeval  # noqa: F401
    except ImportError:
        print("⚠ deepeval chưa cài. Chạy: pip install deepeval")
        return

    dataset = load_golden_dataset()
    print(f"Loaded {len(dataset)} test cases từ golden_dataset.json")

    config_keys = ["A", "B"] if args.config == "both" else [args.config]
    results = {}
    for k in config_keys:
        results[k] = evaluate_config(k, dataset, limit=args.limit)

    # In tóm tắt
    print(f"\n{'='*70}")
    print("  TÓM TẮT ĐIỂM TRUNG BÌNH")
    print(f"{'='*70}")
    for k in config_keys:
        avgs = results[k]["averages"]
        print(f"  Config {k} ({CONFIGS[k]['name']}): overall = {avgs['overall']:.3f}")
        for m in METRIC_NAMES:
            print(f"      {m:22s}: {avgs[m]:.3f}")

    export_results(results)


if __name__ == "__main__":
    main()
