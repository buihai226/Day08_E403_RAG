"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Hỗ trợ:
    - .txt  → wrap text trong markdown (legal docs đã crawl dạng plain text)
    - .json → parse metadata + content, tạo markdown với YAML-style header
    - .pdf / .docx → dùng MarkItDown của Microsoft (nếu có file dạng này)

Output: data/standardized/{legal,news}/*.md

Cài đặt:
    pip install markitdown   (chỉ cần nếu có file PDF/DOCX)
"""

import json
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONVERT LEGAL DOCUMENTS
# Các file pháp luật đã crawl ở Task 1 đều là .txt (plain text).
# Nếu có thêm file PDF/DOCX thì dùng MarkItDown để convert.
# =============================================================================

def convert_legal_docs():
    """Convert tất cả file trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not legal_dir.exists():
        print("  ⚠ Thư mục data/landing/legal/ không tồn tại.")
        return

    # Dedup theo stem: 1 văn bản có thể tồn tại nhiều định dạng (.txt/.docx/.pdf).
    # Chỉ convert 1 lần, ưu tiên nguồn sạch nhất để tránh ghi đè md bằng bản xấu.
    # Ưu tiên: .txt > .docx > .doc > .pdf
    PRIORITY = {".txt": 0, ".docx": 1, ".doc": 2, ".pdf": 3}
    best_by_stem: dict[str, "Path"] = {}
    for f in legal_dir.iterdir():
        if not f.is_file() or f.suffix.lower() not in PRIORITY:
            continue  # bỏ .py, .gitkeep, định dạng khác
        cur = best_by_stem.get(f.stem)
        if cur is None or PRIORITY[f.suffix.lower()] < PRIORITY[cur.suffix.lower()]:
            best_by_stem[f.stem] = f

    for filepath in sorted(best_by_stem.values(), key=lambda p: p.name):
        output_path = output_dir / f"{filepath.stem}.md"

        # --- Plain text (.txt) → Markdown ---
        if filepath.suffix.lower() == ".txt":
            print(f"  📄 Converting TXT: {filepath.name}")
            text = filepath.read_text(encoding="utf-8")

            # Tạo markdown: thêm tiêu đề từ tên file, giữ nguyên nội dung
            # Tên file dạng "luat-phong-chong-ma-tuy-2025.txt"
            # → Title: "Luật phòng chống ma túy 2025"
            title = filepath.stem.replace("-", " ").title()
            markdown = f"# {title}\n\n"
            markdown += f"**Nguồn:** thuvienphapluat.vn\n"
            markdown += f"**File gốc:** {filepath.name}\n\n"
            markdown += "---\n\n"
            markdown += text

            output_path.write_text(markdown, encoding="utf-8")
            print(f"    ✅ → {output_path.name} ({output_path.stat().st_size:,} bytes)")

        # --- PDF / DOCX → MarkItDown ---
        elif filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"  📄 Converting {filepath.suffix.upper()}: {filepath.name}")
            try:
                from markitdown import MarkItDown
                md = MarkItDown()
                result = md.convert(str(filepath))
                output_path.write_text(result.text_content, encoding="utf-8")
                print(f"    ✅ → {output_path.name}")
            except ImportError:
                print("    ⚠ markitdown chưa cài. Chạy: pip install markitdown")
            except Exception as e:
                print(f"    ❌ Lỗi convert: {e}")

        else:
            print(f"  ⏭ Bỏ qua: {filepath.name} (không hỗ trợ)")


# =============================================================================
# CONVERT NEWS ARTICLES
# Các bài báo đã crawl ở Task 2 lưu dạng JSON với fields:
#   url, title, date_crawled, content_markdown
# =============================================================================

def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not news_dir.exists():
        print("  ⚠ Thư mục data/landing/news/ không tồn tại.")
        return

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() != ".json":
            print(f"  ⏭ Bỏ qua: {filepath.name}")
            continue

        print(f"  📰 Converting JSON: {filepath.name}")

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON không hợp lệ: {e}")
            continue

        # Trích xuất metadata
        title = data.get("title", "Không có tiêu đề")
        url = data.get("url", "N/A")
        date_crawled = data.get("date_crawled", "N/A")
        content = data.get("content_markdown", "")

        # Tạo markdown với metadata header
        markdown = f"# {title}\n\n"
        markdown += f"**URL:** {url}\n"
        markdown += f"**Ngày crawl:** {date_crawled}\n\n"
        markdown += "---\n\n"
        markdown += content

        output_path = output_dir / f"{filepath.stem}.md"
        output_path.write_text(markdown, encoding="utf-8")
        print(f"    ✅ → {output_path.name} ({output_path.stat().st_size:,} bytes)")


# =============================================================================
# MAIN — Convert toàn bộ
# =============================================================================

def convert_all():
    """Convert toàn bộ files trong data/landing/ → data/standardized/."""
    print("=" * 60)
    print("Task 3: Convert to Markdown")
    print(f"  Input:  {LANDING_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    # Tổng kết
    print("\n" + "=" * 60)
    print("Tổng kết:")
    md_files = list(OUTPUT_DIR.rglob("*.md"))
    for f in md_files:
        print(f"  📝 {f.relative_to(OUTPUT_DIR)} ({f.stat().st_size:,} bytes)")
    print(f"  Tổng: {len(md_files)} file(s)")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    convert_all()
