"""
Task 10 — Generation Có Citation.

Pipeline end-to-end:
    1. Retrieve relevant chunks (Task 9)
    2. Reorder để tránh "lost in the middle" effect
    3. Format context với source labels
    4. Build prompt (system + context + query)
    5. Call LLM (hỗ trợ cả OpenAI và Google Gemini)
    6. Return answer + sources

"Lost in the middle" (Liu et al., 2023):
    LLM nhớ tốt thông tin ở ĐẦU và CUỐI context, quên thông tin ở GIỮA.
    → Đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

Cài đặt:
    pip install openai python-dotenv
    # hoặc: pip install google-generativeai
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Thêm project root vào path
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k = 5: Số chunks đưa vào context
# Vì sao 5? Đủ evidence (3-5 nguồn) để trả lời câu hỏi pháp luật,
# nhưng không quá nhiều gây "lost in the middle" hoặc vượt context window.
# Nghiên cứu cho thấy LLM xử lý tốt nhất với 3-7 chunks.
TOP_K = 5

# top_p = 0.9 (nucleus sampling): Chỉ sample từ top 90% xác suất tích luỹ
# Vì sao 0.9? Đủ diverse để câu trả lời tự nhiên, nhưng không quá random.
# Giá trị thấp hơn (0.5) → quá cứng nhắc; cao hơn (0.99) → có thể hallucinate.
TOP_P = 0.9

# temperature = 0.3: Độ ngẫu nhiên thấp
# Vì sao 0.3? RAG cần factual accuracy — câu trả lời pháp luật phải CHÍNH XÁC.
# Temperature thấp giúp model ưu tiên token có xác suất cao nhất (ít sáng tạo).
# 0.0 = deterministic, 0.3 = ít creative, 0.7 = balanced, 1.0 = creative
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT — Hướng dẫn LLM trả lời có citation
# =============================================================================

SYSTEM_PROMPT = """Bạn là trợ lý pháp luật chuyên về luật ma túy tại Việt Nam.
Hãy trả lời câu hỏi dựa HOÀN TOÀN trên context được cung cấp.

Quy tắc:
1. Mỗi phát biểu về sự kiện PHẢI có citation trong ngoặc vuông, ví dụ:
   [Luật Phòng chống ma túy 2025, Điều 3] hoặc [VnExpress, 2024]
2. Nếu thông tin KHÔNG có trong context → nói rõ:
   "Tôi không thể xác minh thông tin này từ nguồn hiện có"
3. Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc
4. Trích dẫn số điều luật cụ thể nếu có
5. Phân biệt rõ giữa luật pháp và tin tức báo chí"""


# =============================================================================
# DOCUMENT REORDERING — Tránh "Lost in the Middle"
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    Paper: Liu et al. (2023) "Lost in the Middle: How Language Models
           Use Long Contexts"

    Phát hiện: LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt,
    nhưng quên/bỏ sót thông tin ở GIỮA.

    Strategy: Interleave — đặt chunks quan trọng ở đầu và cuối:
        Input (by score):  [1, 2, 3, 4, 5]  (1 = relevant nhất)
        Output:            [1, 3, 5, 4, 2]  (best first, worst middle)

    Cách làm:
        - Vị trí lẻ (1, 3, 5) → đặt ở đầu (theo thứ tự)
        - Vị trí chẵn (2, 4) → đặt ở cuối (đảo ngược)

    Args:
        chunks: List sorted by score descending (từ retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    reordered = []

    # Lấy các chunks ở vị trí lẻ (0, 2, 4...) → đặt ở đầu
    for i in range(0, len(chunks), 2):
        reordered.append(chunks[i])

    # Lấy các chunks ở vị trí chẵn (1, 3, 5...) → đặt ở cuối (đảo ngược)
    even_chunks = [chunks[i] for i in range(1, len(chunks), 2)]
    reordered.extend(reversed(even_chunks))

    return reordered


# =============================================================================
# CONTEXT FORMATTING — Tạo context string cho prompt
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source rõ ràng để LLM có thể cite.

    Ví dụ output:
        [Document 1 | Source: luat-phong-chong-ma-tuy-2025.md | Type: legal]
        Điều 3. Giải thích từ ngữ...
        ---
        [Document 2 | Source: article_01.md | Type: news]
        Ca sĩ Chi Dân bị bắt...

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", f"Source {i}")
        doc_type = chunk.get("metadata", {}).get("type", "unknown")
        score = chunk.get("score", 0)

        # Header cho mỗi document — LLM sẽ dùng info này để cite
        header = f"[Document {i} | Source: {source} | Type: {doc_type} | Score: {score:.3f}]"
        context_parts.append(f"{header}\n{chunk['content']}")

    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# LLM CALL — Hỗ trợ OpenAI và Google Gemini
# =============================================================================

def _call_openai(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI API (GPT-4o-mini hoặc GPT-3.5-turbo)."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Rẻ, nhanh, đủ tốt cho RAG
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=2000,
    )
    return response.choices[0].message.content


def _call_gemini(system_prompt: str, user_message: str) -> str:
    """Gọi Google Gemini API (fallback nếu không có OpenAI key)."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Gemini không có system prompt riêng — gộp vào user message
    full_prompt = f"{system_prompt}\n\n{user_message}"
    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=TEMPERATURE,
            top_p=TOP_P,
            max_output_tokens=2000,
        ),
    )
    return response.text


def _call_llm(system_prompt: str, user_message: str) -> str:
    """
    Gọi LLM — tự động chọn provider dựa trên API key có sẵn.
    Thứ tự ưu tiên: OpenAI → Gemini → Error message.
    """
    # Thử OpenAI trước
    if os.getenv("OPENAI_API_KEY"):
        try:
            return _call_openai(system_prompt, user_message)
        except Exception as e:
            print(f"  ⚠ OpenAI failed: {e}")

    # Fallback: Google Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            return _call_gemini(system_prompt, user_message)
        except Exception as e:
            print(f"  ⚠ Gemini failed: {e}")

    # Không có API key nào
    return (
        "⚠ Không thể gọi LLM — chưa có API key.\n"
        "Hãy thêm OPENAI_API_KEY hoặc GOOGLE_API_KEY vào file .env\n\n"
        "--- Context đã retrieve ---\n"
        "(Xem phần sources bên dưới để biết các chunks đã tìm được)"
    )


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def _format_history(history: list[dict] | None, max_turns: int = 4) -> str:
    """
    Format lịch sử hội thoại gần nhất thành text để đưa vào prompt.
    Chỉ lấy max_turns lượt cuối để tránh prompt quá dài.
    history: list of {'role': 'user'/'assistant', 'content': str}
    """
    if not history:
        return ""
    recent = history[-max_turns * 2:]  # mỗi turn = user + assistant
    lines = []
    for m in recent:
        role = "Người dùng" if m.get("role") == "user" else "Trợ lý"
        content = (m.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    if not lines:
        return ""
    return "Lịch sử hội thoại trước đó:\n" + "\n".join(lines) + "\n\n---\n\n"


def generate_with_citation(
    query: str, top_k: int = TOP_K, history: list[dict] | None = None
) -> dict:
    """
    End-to-end RAG generation có citation + hỗ trợ hội thoại nhiều lượt.

    Pipeline:
        1. Retrieve relevant chunks (semantic + lexical + rerank)
        2. Reorder chunks (lost in the middle prevention)
        3. Format context với source labels
        4. Build prompt (system + history + context + query)
        5. Call LLM (OpenAI hoặc Gemini)
        6. Return answer + sources

    Args:
        query: Câu hỏi của user (tiếng Việt)
        top_k: Số chunks đưa vào context
        history: (tùy chọn) lịch sử hội thoại để hỗ trợ follow-up questions,
                 list of {'role': 'user'/'assistant', 'content': str}

    Returns:
        {
            'answer': str,
            'sources': list[dict],
            'retrieval_source': str
        }
    """
    print(f"\n  🔍 Retrieving context for: {query}")

    # Step 1: Retrieve relevant chunks
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "Tôi không tìm thấy thông tin liên quan trong cơ sở dữ liệu.",
            "sources": [],
            "retrieval_source": "none",
        }

    print(f"  📚 Retrieved {len(chunks)} chunks")

    # Step 2: Reorder — đặt chunks quan trọng ở đầu và cuối
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context với source labels
    context = format_context(reordered)

    # Step 4: Build prompt (kèm lịch sử hội thoại nếu có)
    history_block = _format_history(history)
    user_message = (
        f"{history_block}"
        f"Context:\n{context}\n\n"
        f"---\n\n"
        f"Câu hỏi: {query}"
    )

    # Step 5: Call LLM
    print(f"  🤖 Generating answer...")
    answer = _call_llm(SYSTEM_PROMPT, user_message)

    # Step 6: Return
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
    }


# =============================================================================
# INTERACTIVE CHAT — Tự gõ câu hỏi trong terminal
# =============================================================================

def interactive_chat():
    """
    Chat tương tác với RAG trong terminal.
    Gõ câu hỏi → nhận câu trả lời có citation + nguồn.
    Gõ 'exit' / 'quit' / 'q' để thoát.
    """
    print("=" * 70)
    print("  RAG Chatbot — Pháp luật & tin tức về ma túy (Task 10)")
    print("=" * 70)
    print("  Gõ câu hỏi rồi Enter. Gõ 'exit' / 'quit' / 'q' để thoát.")
    print("=" * 70)

    history: list[dict] = []
    while True:
        try:
            query = input("\n🧑 Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Tạm biệt!")
            break

        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            print("👋 Tạm biệt!")
            break

        try:
            result = generate_with_citation(query, history=history)
        except Exception as e:
            print(f"  ❌ Lỗi khi sinh câu trả lời: {e}")
            continue

        print(f"\n🤖 Trợ lý:\n{result['answer']}")
        print(f"\n📎 Nguồn: {len(result['sources'])} chunks "
              f"| via {result['retrieval_source']}")
        for i, s in enumerate(result["sources"], 1):
            src = s.get("metadata", {}).get("source", "?")
            doc_type = s.get("metadata", {}).get("type", "?")
            print(f"   {i}. [{s['score']:.3f}] ({doc_type}) {src}")

        # Lưu vào lịch sử để hỗ trợ câu hỏi follow-up
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    interactive_chat()
