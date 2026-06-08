import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re

# ============================================================
# Crawl & clean text: Luật Phòng, chống ma túy 2025
# URL: https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/
#      Luat-Phong-chong-ma-tuy-2025-so-120-2025-QH15-666019.aspx
# ============================================================

URL = (
    "https://thuvienphapluat.vn/van-ban/Van-hoa-Xa-hoi/"
    "Luat-Phong-chong-ma-tuy-2025-so-120-2025-QH15-666019.aspx"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    )
}

OUTPUT_FILE = Path(__file__).parent / "luat_phong_chong_ma_tuy_2025.txt"


def fetch_html(url: str) -> str:
    """Tải HTML từ URL."""
    res = requests.get(url, headers=HEADERS, timeout=30)
    res.raise_for_status()
    res.encoding = "utf-8"
    return res.text


def extract_law_content(html: str) -> str:
    """
    Trích xuất nội dung luật từ div.content1.
    Loại bỏ hoàn toàn: script, style, noscript, nav, header, footer,
    form, iframe, img, input, select, button, và các thẻ không cần thiết.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Tìm div chứa nội dung luật (class="content1")
    content_div = soup.find("div", class_="content1")
    if not content_div:
        raise ValueError(
            "Không tìm thấy div.content1 — cấu trúc trang có thể đã thay đổi."
        )

    # Xóa các thẻ không mong muốn bên trong content_div
    tags_to_remove = [
        "script", "style", "noscript", "nav", "header", "footer",
        "form", "iframe", "img", "input", "select", "button",
        "textarea", "svg", "canvas",
    ]
    for tag in content_div.find_all(tags_to_remove):
        tag.decompose()

    # Thay <br> bằng newline trước khi xử lý
    for br in content_div.find_all("br"):
        br.replace_with("\n")

    # Trích xuất text theo từng block-level element,
    # nối inline text trong mỗi block để tránh ngắt dòng giữa câu.
    block_tags = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
                  "tr", "td", "th", "li", "blockquote", "pre",
                  "table", "section", "article"}

    paragraphs = []
    for element in content_div.descendants:
        if element.name in block_tags:
            # Lấy text trực tiếp (không đệ quy sâu vào block con)
            text_parts = []
            for child in element.children:
                if child.name and child.name in block_tags:
                    continue  # Bỏ qua block con, sẽ xử lý riêng
                child_text = child.get_text(" ") if child.name else str(child)
                text_parts.append(child_text)
            joined = " ".join(text_parts)
            # Gộp khoảng trắng
            joined = re.sub(r"\s+", " ", joined).strip()
            if joined:
                paragraphs.append(joined)

    raw_text = "\n".join(paragraphs)
    return raw_text


def clean_text(raw: str) -> str:
    """
    Làm sạch text:
    - Chuẩn hóa line endings
    - Strip khoảng trắng thừa mỗi dòng
    - Loại bỏ dòng trống thừa (chỉ giữ tối đa 1 dòng trống giữa các đoạn)
    - Gộp khoảng trắng ngang thừa
    - Loại bỏ các chuỗi UI rác (nếu còn sót)
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Strip từng dòng và loại bỏ dòng trống
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            # Gộp khoảng trắng ngang thừa
            stripped = re.sub(r"[ \t]+", " ", stripped)
            lines.append(stripped)

    text = "\n".join(lines)

    # --- Cắt lấy phần nội dung luật chính ---
    # Bắt đầu: "QUỐC HỘI"
    start_markers = ["QUỐC HỘI", "Luật số: 120/2025/QH15"]
    start = -1
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            if start == -1 or pos < start:
                start = pos

    if start == -1:
        raise ValueError("Không tìm thấy điểm bắt đầu văn bản luật.")

    text = text[start:]

    # Kết thúc: sau tên người ký "Trần Thanh Mẫn"
    end_markers = ["Trần Thanh Mẫn"]
    end = -1
    for marker in end_markers:
        pos = text.rfind(marker)  # rfind để lấy lần xuất hiện cuối
        if pos != -1:
            end = pos + len(marker)

    if end == -1:
        # Fallback: cắt đến "CHỦ TỊCH QUỐC HỘI"
        pos = text.rfind("CHỦ TỊCH QUỐC HỘI")
        if pos != -1:
            end = pos + len("CHỦ TỊCH QUỐC HỘI")

    if end != -1:
        text = text[:end]

    # --- Loại bỏ các chuỗi UI / rác còn sót ---
    ui_noise_patterns = [
        r"Đăng nhập.*",
        r"Đăng ký.*",
        r"Đăng xuất.*",
        r"Quên mật khẩu.*",
        r"Tìm kiếm.*",
        r"Tải về.*",
        r"In văn bản.*",
        r"Gửi văn bản.*",
        r"Chia sẻ.*",
        r"Lưu vào.*",
        r"Thêm vào.*",
        r"Xem thêm.*tại đây.*",
        r"Click vào.*để xem.*",
        r"MỤC LỤC VĂN BẢN",
        r"In mục lục",
        r"\xa0",  # non-breaking space đơn lẻ
    ]
    for pattern in ui_noise_patterns:
        text = re.sub(pattern, "", text)

    # Loại bỏ chuỗi base64 rác từ HTML comments (vd: VABWAFAATABf...)
    text = re.sub(r"[A-Za-z0-9+/]{20,}={0,2}", "", text)

    # Loại dòng chỉ chứa dấu gạch ngang trang trí (-------)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)

    # Xóa dấu gạch ngang trang trí nằm trong dòng (vd: "QUỐC HỘI -------")
    text = re.sub(r"\s*-{3,}\s*", "\n", text)

    # Fix khoảng trắng thừa trước dấu câu
    text = re.sub(r"\s+([;.,:])", r"\1", text)

    # Xóa dòng trống thừa (chỉ giữ 1 dòng trống)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Xóa khoảng trắng đầu/cuối toàn bộ
    text = text.strip()

    return text


def main():
    print(f"Đang crawl dữ liệu từ:\n  {URL}\n")

    html = fetch_html(URL)
    print(f"  → Tải HTML thành công ({len(html):,} bytes)")

    raw_content = extract_law_content(html)
    print(f"  → Trích xuất nội dung từ div.content1 ({len(raw_content):,} ký tự)")

    clean = clean_text(raw_content)
    print(f"  → Clean text hoàn tất ({len(clean):,} ký tự)")

    OUTPUT_FILE.write_text(clean, encoding="utf-8")
    print(f"\n✅ Đã lưu: {OUTPUT_FILE}")
    print(f"   Dung lượng: {OUTPUT_FILE.stat().st_size:,} bytes")

    # Preview
    print("\n--- Preview (500 ký tự đầu) ---")
    print(clean[:500])
    print("\n--- Preview (500 ký tự cuối) ---")
    print(clean[-500:])


if __name__ == "__main__":
    main()