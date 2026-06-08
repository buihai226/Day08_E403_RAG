"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Crawl tối thiểu 3 văn bản pháp luật từ thuvienphapluat.vn
    2. Lưu clean text vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.

Nguồn: https://thuvienphapluat.vn
"""

import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

# ============================================================
# Danh sách 3 văn bản pháp luật cần crawl
# ============================================================
LEGAL_DOCS = [
    {
        "url": (
            "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/"
            "Luat-Phong-chong-ma-tuy-2025-so-120-2025-QH15-666019.aspx"
        ),
        "filename": "luat-phong-chong-ma-tuy-2025.txt",
        "description": "Luật Phòng, chống ma túy 2025 (120/2025/QH15)",
    },
    {
        "url": (
            "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/"
            "Nghi-dinh-28-2026-ND-CP-Danh-muc-chat-ma-tuy-va-tien-chat-690473.aspx"
        ),
        "filename": "nghi-dinh-28-2026-danh-muc-chat-ma-tuy.txt",
        "description": "Nghị định 28/2026/NĐ-CP về Danh mục chất ma túy và tiền chất",
    },
    {
        "url": (
            "https://thuvienphapluat.vn/van-ban/Trach-nhiem-hinh-su/"
            "Bo-luat-hinh-su-2015-296661.aspx"
        ),
        "filename": "bo-luat-hinh-su-2015-chuong-xx-ma-tuy.txt",
        "description": "Bộ luật Hình sự 2015 — Chương XX: Các tội phạm về ma túy",
    },
]


# ============================================================
# Các hàm tiện ích
# ============================================================

def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_html(url: str) -> str:
    """Tải HTML từ URL."""
    res = requests.get(url, headers=HEADERS, timeout=30)
    res.raise_for_status()
    res.encoding = "utf-8"
    return res.text


def extract_law_content(html: str) -> str:
    """
    Trích xuất nội dung văn bản pháp luật từ div.content1.
    Loại bỏ các thẻ HTML không cần thiết, nối inline text trong mỗi
    block-level element để tránh ngắt dòng giữa câu.
    """
    soup = BeautifulSoup(html, "html.parser")

    content_div = soup.find("div", class_="content1")
    if not content_div:
        raise ValueError(
            "Không tìm thấy div.content1 — cấu trúc trang có thể đã thay đổi."
        )

    # Xóa các thẻ không mong muốn
    tags_to_remove = [
        "script", "style", "noscript", "nav", "header", "footer",
        "form", "iframe", "img", "input", "select", "button",
        "textarea", "svg", "canvas",
    ]
    for tag in content_div.find_all(tags_to_remove):
        tag.decompose()

    # Thay <br> bằng newline
    for br in content_div.find_all("br"):
        br.replace_with("\n")

    # Trích xuất text theo block-level element
    block_tags = {
        "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
        "tr", "td", "th", "li", "blockquote", "pre",
        "table", "section", "article",
    }

    paragraphs = []
    for element in content_div.descendants:
        if element.name in block_tags:
            text_parts = []
            for child in element.children:
                if child.name and child.name in block_tags:
                    continue
                child_text = child.get_text(" ") if child.name else str(child)
                text_parts.append(child_text)
            joined = " ".join(text_parts)
            joined = re.sub(r"\s+", " ", joined).strip()
            if joined:
                paragraphs.append(joined)

    return "\n".join(paragraphs)


def clean_text(raw: str) -> str:
    """
    Làm sạch text:
    - Chuẩn hóa line endings, strip khoảng trắng thừa
    - Tự động phát hiện điểm bắt đầu nội dung (QUỐC HỘI / CHÍNH PHỦ / ...)
    - Tự động phát hiện điểm kết thúc (tên người ký cuối cùng)
    - Loại bỏ UI noise, base64, dấu gạch ngang trang trí
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Strip từng dòng, bỏ dòng trống
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            stripped = re.sub(r"[ \t]+", " ", stripped)
            lines.append(stripped)
    text = "\n".join(lines)

    # --- Tự động tìm điểm bắt đầu văn bản pháp luật ---
    # Các cơ quan ban hành phổ biến
    start_markers = [
        "QUỐC HỘI",
        "CHÍNH PHỦ",
        "THỦ TƯỚNG CHÍNH PHỦ",
        "BỘ CÔNG AN",
        "BỘ Y TẾ",
    ]
    start = -1
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            if start == -1 or pos < start:
                start = pos

    if start == -1:
        # Fallback: tìm "CỘNG HÒA XÃ HỘI"
        start = text.find("CỘNG HÒA XÃ HỘI")

    if start != -1:
        text = text[start:]

    # --- Tự động tìm điểm kết thúc (tên người ký) ---
    # Các chức danh ký phổ biến
    signer_titles = [
        "CHỦ TỊCH QUỐC HỘI",
        "THỦ TƯỚNG",
        "TM. CHÍNH PHỦ",
        "BỘ TRƯỞNG",
        "KT. THỦ TƯỚNG",
    ]

    # Tìm vị trí cuối cùng của các chức danh, rồi lấy dòng tiếp theo
    # (dòng tiếp theo thường là tên người ký)
    end = -1
    for title in signer_titles:
        pos = text.rfind(title)
        if pos != -1:
            # Tìm đến cuối dòng có tên người ký (1-2 dòng sau chức danh)
            remaining = text[pos:]
            remaining_lines = remaining.split("\n")
            # Lấy chức danh + tối đa 3 dòng sau (tên người ký)
            signer_end = pos
            for i, line in enumerate(remaining_lines[:5]):
                signer_end = pos + sum(len(l) + 1 for l in remaining_lines[:i+1])
            candidate_end = min(signer_end, len(text))
            if candidate_end > end:
                end = candidate_end

    if end != -1:
        text = text[:end]

    # --- Loại bỏ UI noise ---
    ui_noise_patterns = [
        r"Đăng nhập.*", r"Đăng ký.*", r"Đăng xuất.*",
        r"Quên mật khẩu.*", r"Tìm kiếm.*", r"Tải về.*",
        r"In văn bản.*", r"Gửi văn bản.*", r"Chia sẻ.*",
        r"Lưu vào.*", r"Thêm vào.*",
        r"Xem thêm.*tại đây.*", r"Click vào.*để xem.*",
        r"MỤC LỤC VĂN BẢN", r"In mục lục",
        r"\xa0",
    ]
    for pattern in ui_noise_patterns:
        text = re.sub(pattern, "", text)

    # Loại bỏ chuỗi base64 rác
    text = re.sub(r"[A-Za-z0-9+/]{20,}={0,2}", "", text)

    # Loại dòng/chuỗi gạch ngang trang trí
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*-{3,}\s*", "\n", text)

    # Fix khoảng trắng thừa trước dấu câu
    text = re.sub(r"\s+([;.,:])", r"\1", text)

    # Xóa dòng trống thừa
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def crawl_one_document(doc: dict) -> None:
    """Crawl và lưu 1 văn bản pháp luật."""
    url = doc["url"]
    filename = doc["filename"]
    description = doc["description"]
    output_path = DATA_DIR / filename

    # Bỏ qua nếu đã crawl
    if output_path.exists() and output_path.stat().st_size > 1000:
        print(f"  ⏭ Đã tồn tại, bỏ qua: {filename}")
        return

    print(f"  📥 Đang tải: {description}")
    print(f"     URL: {url}")

    html = fetch_html(url)
    print(f"     → HTML: {len(html):,} bytes")

    raw_content = extract_law_content(html)
    print(f"     → Trích xuất: {len(raw_content):,} ký tự")

    clean = clean_text(raw_content)
    print(f"     → Clean text: {len(clean):,} ký tự")

    output_path.write_text(clean, encoding="utf-8")
    print(f"     ✅ Đã lưu: {output_path.name} ({output_path.stat().st_size:,} bytes)")


def main():
    """Crawl tất cả văn bản pháp luật trong danh sách."""
    setup_directory()

    print("=" * 60)
    print("Task 1 — Thu thập văn bản pháp luật về ma tuý")
    print(f"  Số văn bản: {len(LEGAL_DOCS)}")
    print(f"  Thư mục: {DATA_DIR}")
    print("=" * 60)

    # Nếu file cũ đã crawl bằng tên khác, rename cho đúng format
    old_file = DATA_DIR / "luat_phong_chong_ma_tuy_2025.txt"
    new_file = DATA_DIR / "luat-phong-chong-ma-tuy-2025.txt"
    if old_file.exists() and not new_file.exists():
        old_file.rename(new_file)
        print(f"  🔄 Đổi tên: {old_file.name} → {new_file.name}")

    for i, doc in enumerate(LEGAL_DOCS, 1):
        print(f"\n[{i}/{len(LEGAL_DOCS)}] {doc['description']}")
        crawl_one_document(doc)

    # Tổng kết
    print("\n" + "=" * 60)
    print("Tổng kết:")
    txt_files = list(DATA_DIR.glob("*.txt"))
    for f in txt_files:
        print(f"  📄 {f.name} ({f.stat().st_size:,} bytes)")
    print(f"  Tổng: {len(txt_files)} file(s)")
    print("=" * 60)


if __name__ == "__main__":
    main()
