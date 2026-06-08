"""
app.py — RAG Chatbot UI (Streamlit) cho dự án "Pháp luật & tin tức về ma túy".

Mục tiêu: cho phép chat với hệ thống RAG VÀ nhìn thấy TỪNG BƯỚC của pipeline
mỗi khi search:

    Query
      ├─ Bước 1: Semantic Search (dense / vector)        [Task 5]
      ├─ Bước 2: Lexical Search (BM25)                   [Task 6]
      ├─ Bước 3: Merge bằng RRF                          [Task 7]
      ├─ Bước 4: Rerank                                  [Task 7]
      ├─ Bước 5: Kiểm tra ngưỡng → Fallback PageIndex    [Task 8/9]
      ├─ Bước 6: Reorder chống "lost in the middle"      [Task 10]
      └─ Bước 7: Generation có citation (LLM)            [Task 10]

Chạy:
    streamlit run app.py
"""

import sys
import time
from pathlib import Path

import streamlit as st

# Đảm bảo import được package src/
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các module pipeline (đặt trong hàm cache để không load lại mỗi rerun)
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf, rerank
from src.task8_pageindex_vectorless import pageindex_search
from src.task10_generation import (
    reorder_for_llm,
    format_context,
    _call_llm,
    SYSTEM_PROMPT,
)


# =============================================================================
# CẤU HÌNH TRANG
# =============================================================================

st.set_page_config(
    page_title="RAG Chatbot — Ma túy & Pháp luật VN",
    page_icon="⚖️",
    layout="wide",
)


# =============================================================================
# HÀM TIỆN ÍCH HIỂN THỊ
# =============================================================================

def render_results(results: list[dict], show_full: bool = False):
    """Hiển thị danh sách kết quả retrieval dưới dạng bảng gọn."""
    if not results:
        st.info("Không có kết quả ở bước này.")
        return

    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        src = meta.get("source", "?")
        doc_type = meta.get("type", "?")
        score = r.get("score", 0)
        via = r.get("source", "")

        badge = "⚖️ legal" if doc_type == "legal" else "📰 news"
        via_txt = f" · via `{via}`" if via else ""
        header = f"**{i}.** `{score:.4f}` · {badge} · `{src}`{via_txt}"
        st.markdown(header)

        content = r.get("content", "")
        snippet = content if show_full else (content[:220] + ("..." if len(content) > 220 else ""))
        st.caption(snippet.replace("\n", " "))


# =============================================================================
# PIPELINE CÓ HIỂN THỊ TỪNG BƯỚC
# =============================================================================

def run_pipeline_with_steps(
    query: str,
    top_k: int,
    score_threshold: float,
    rerank_method: str,
    use_reranking: bool,
    use_fallback: bool,
):
    """
    Chạy pipeline retrieval + generation, hiển thị từng bước trong UI.

    Returns:
        (answer: str, final_results: list[dict], retrieval_source: str)
    """
    timings = {}

    # ---- Bước 1: Semantic Search ----
    with st.status("🔎 Bước 1 — Semantic Search (vector / dense)", expanded=True) as s:
        t0 = time.time()
        dense = semantic_search(query, top_k=top_k * 2)
        timings["semantic"] = time.time() - t0
        st.write(f"Tìm thấy **{len(dense)}** kết quả theo ngữ nghĩa "
                 f"({timings['semantic']:.2f}s)")
        render_results(dense)
        s.update(label=f"✅ Bước 1 — Semantic Search ({len(dense)} kết quả)",
                 state="complete", expanded=False)

    # ---- Bước 2: Lexical Search (BM25) ----
    with st.status("🔠 Bước 2 — Lexical Search (BM25)", expanded=True) as s:
        t0 = time.time()
        sparse = lexical_search(query, top_k=top_k * 2)
        timings["lexical"] = time.time() - t0
        st.write(f"Tìm thấy **{len(sparse)}** kết quả theo từ khóa "
                 f"({timings['lexical']:.2f}s)")
        render_results(sparse)
        s.update(label=f"✅ Bước 2 — Lexical Search ({len(sparse)} kết quả)",
                 state="complete", expanded=False)

    # ---- Bước 3: Merge bằng RRF ----
    with st.status("🔀 Bước 3 — Merge (Reciprocal Rank Fusion)", expanded=True) as s:
        merged = rerank_rrf(ranked_lists=[dense, sparse], top_k=top_k * 2)
        for item in merged:
            item["source"] = "hybrid"
        st.write(f"Gộp 2 danh sách → **{len(merged)}** kết quả (document xuất hiện "
                 f"ở cả 2 nguồn được xếp hạng cao hơn).")
        render_results(merged)
        s.update(label=f"✅ Bước 3 — RRF Merge ({len(merged)} kết quả)",
                 state="complete", expanded=False)

    # ---- Bước 4: Rerank ----
    if use_reranking and merged:
        with st.status(f"📊 Bước 4 — Rerank (method = {rerank_method})", expanded=True) as s:
            final_results = rerank(query, merged, top_k=top_k, method=rerank_method)
            st.write(f"Xếp hạng lại → giữ top **{len(final_results)}**.")
            render_results(final_results)
            s.update(label=f"✅ Bước 4 — Rerank ({len(final_results)} kết quả)",
                     state="complete", expanded=False)
    else:
        final_results = merged[:top_k]
        st.info("⏭ Bỏ qua rerank (tắt trong sidebar).")

    # ---- Bước 5: Kiểm tra ngưỡng → Fallback PageIndex ----
    retrieval_source = "hybrid"
    best_score = final_results[0]["score"] if final_results else 0
    with st.status("🛟 Bước 5 — Kiểm tra ngưỡng & Fallback", expanded=True) as s:
        st.write(f"Score cao nhất = `{best_score:.4f}` · ngưỡng = `{score_threshold}`")
        if use_fallback and (not final_results or best_score < score_threshold):
            st.warning("Score dưới ngưỡng → thử **PageIndex Vectorless** (Task 8).")
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                final_results = fallback
                retrieval_source = "pageindex"
                st.write(f"PageIndex trả về **{len(fallback)}** kết quả.")
                render_results(fallback)
            else:
                st.write("PageIndex không có kết quả → giữ kết quả hybrid.")
        else:
            st.success("Score đạt ngưỡng → dùng kết quả hybrid, không cần fallback.")
        s.update(label=f"✅ Bước 5 — Nguồn cuối: {retrieval_source}",
                 state="complete", expanded=False)

    if not final_results:
        return ("Không tìm thấy thông tin liên quan trong cơ sở dữ liệu.",
                [], "none")

    # ---- Bước 6: Reorder chống lost-in-the-middle ----
    with st.status("🔄 Bước 6 — Reorder chống 'lost in the middle'", expanded=True) as s:
        reordered = reorder_for_llm(final_results)
        order_txt = " → ".join(
            f"{r.get('metadata', {}).get('source', '?')}" for r in reordered
        )
        st.write("Thứ tự đưa vào LLM (quan trọng ở đầu & cuối):")
        st.caption(order_txt)
        s.update(label="✅ Bước 6 — Reorder xong", state="complete", expanded=False)

    # ---- Bước 7: Generation ----
    with st.status("🤖 Bước 7 — Generation (LLM, có citation)", expanded=True) as s:
        context = format_context(reordered)
        user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"
        with st.expander("Xem context gửi vào LLM"):
            st.text(context)
        t0 = time.time()
        answer = _call_llm(SYSTEM_PROMPT, user_message)
        timings["llm"] = time.time() - t0
        st.write(f"LLM phản hồi ({timings['llm']:.2f}s)")
        s.update(label="✅ Bước 7 — Đã sinh câu trả lời",
                 state="complete", expanded=False)

    return answer, final_results, retrieval_source


# =============================================================================
# SIDEBAR — CẤU HÌNH
# =============================================================================

with st.sidebar:
    st.header("⚙️ Cấu hình pipeline")
    top_k = st.slider("top_k (số chunk cuối)", 1, 10, 5)
    score_threshold = st.slider("Ngưỡng fallback PageIndex", 0.0, 1.0, 0.3, 0.05)
    rerank_method = st.selectbox(
        "Phương pháp rerank",
        ["rrf", "mmr", "cross_encoder"],
        index=0,
        help="rrf: không cần API · cross_encoder: cần JINA_API_KEY",
    )
    use_reranking = st.checkbox("Bật rerank (Bước 4)", value=True)
    use_fallback = st.checkbox("Bật fallback PageIndex (Bước 5)", value=False,
                               help="Cần PAGEINDEX_API_KEY trong .env")
    show_steps = st.checkbox("Hiển thị từng bước pipeline", value=True)

    st.divider()
    st.caption("Nguồn dữ liệu: văn bản pháp luật (legal) + bài báo nghệ sĩ (news).")
    if st.button("🗑 Xóa lịch sử chat"):
        st.session_state.messages = []
        st.rerun()


# =============================================================================
# GIAO DIỆN CHAT CHÍNH
# =============================================================================

st.title("⚖️ RAG Chatbot — Pháp luật & tin tức ma túy VN")
st.caption("Hỏi về luật phòng chống ma túy, hình phạt, hoặc các vụ nghệ sĩ liên quan ma túy.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} nguồn tham khảo"):
                render_results(msg["sources"])

# Ô nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if show_steps:
            st.markdown("#### 🧭 Luồng xử lý của hệ thống")
            answer, sources, retrieval_source = run_pipeline_with_steps(
                prompt, top_k, score_threshold, rerank_method,
                use_reranking, use_fallback,
            )
        else:
            with st.spinner("Đang xử lý..."):
                from src.task10_generation import generate_with_citation
                result = generate_with_citation(prompt, top_k=top_k)
                answer = result["answer"]
                sources = result["sources"]
                retrieval_source = result["retrieval_source"]

        st.markdown("#### 💬 Câu trả lời")
        st.markdown(answer)
        st.caption(f"Nguồn retrieval: **{retrieval_source}** · {len(sources)} chunks")
        with st.expander(f"📎 {len(sources)} nguồn tham khảo"):
            render_results(sources, show_full=False)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
