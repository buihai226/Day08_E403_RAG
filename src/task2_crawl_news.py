"""
Task 2 — Crawl bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng requests + BeautifulSoup (ổn định trên Windows, không cần browser).
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install requests beautifulsoup4
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Danh sách 5 bài báo về nghệ sĩ Việt Nam liên quan tới ma túy
#
# Nếu URL nào bị lỗi 404 khi chạy, hãy thay bằng URL khác.
# Gợi ý tìm kiếm: "nghệ sĩ Việt Nam ma túy" trên các trang
# vnexpress.net, tuoitre.vn, thanhnien.vn, dantri.com.vn
# ============================================================
#
# CÁC URL DƯỚI ĐÂY ĐÃ ĐƯỢC KIỂM CHỨNG (cào được nội dung TIẾNG VIỆT thật).
# Dùng kenh14.vn — báo tiếng Việt, render sẵn HTML + có JSON-LD nên cào ổn định.
# Tất cả là bài về nghệ sĩ Việt Nam liên quan ma túy (Chi Dân, An Tây,
# Long Nhật, Miu Lê, Hữu Tín, Nguyễn Công Trí...).
# Dùng tiếng Việt để khớp với query tiếng Việt + corpus pháp luật tiếng Việt.
#
ARTICLE_URLS = [
    # 1. Bữa tiệc ma túy của Chi Dân + người mẫu An Tây
    "https://kenh14.vn/lo-dien-nhung-bua-tiec-ma-tuy-cua-ca-si-chi-dan-nguoi-mau-an-tay-215260402194539945.chn",
    # 2. Chi Dân và Andrea bị điều tra vì nghi liên quan ma túy
    "https://kenh14.vn/chi-dan-va-andrea-bi-dieu-tra-vi-nghi-lien-quan-ma-tuy-215241110163031423.chn",
    # 3. Công an TP HCM kết luận vụ ca sĩ Chi Dân dùng ma túy
    "https://kenh14.vn/cong-an-tp-hcm-ket-luan-vu-ca-si-chi-dan-dung-ma-tuy-215250821144601372.chn",
    # 4. An Tây rủ trợ lý sử dụng ma túy như thế nào
    "https://kenh14.vn/an-tay-ru-tro-ly-van-anh-duy-su-dung-ma-tuy-nhu-the-nao-215260403140920578.chn",
    # 5. Ca sĩ Long Nhật lên tiếng sau khi bị bắt vì ma túy
    "https://kenh14.vn/ca-si-long-nhat-trong-long-mac-cam-toi-loi-toi-khong-thich-ma-tuy-chut-nao-215260520134900616.chn",
    # 6. Vụ Miu Lê tổ chức, sử dụng ma túy ở đảo Cát Bà
    "https://kenh14.vn/vu-miu-le-to-chuc-su-dung-ma-tuy-o-dao-cat-ba-nhom-6-nguoi-bi-khoi-to-ve-toi-gi-215260516221137139.chn",
    # 7. Diễn viên Hữu Tín lĩnh án 7 năm 6 tháng tù vì ma túy
    "https://kenh14.vn/dien-vien-huu-tin-linh-an-7-nam-6-thang-tu-20230428141120542.chn",
    # 8. Tổng hợp: Sao Việt tiêu tan sự nghiệp vì liên quan ma túy
    "https://kenh14.vn/sao-viet-tieu-tan-su-nghiep-vi-lien-quan-den-ma-tuy-215260522111209355.chn",
]

# Số bài tối thiểu cần crawl thành công (theo yêu cầu Task 2)
MIN_ARTICLES_REQUIRED = 5

# Độ dài nội dung tối thiểu để coi là bài hợp lệ (chống cào trúng trang chủ/rỗng)
MIN_CONTENT_CHARS = 400

# Các tiêu đề "trang chủ" — nếu title rơi vào đây nghĩa là URL đã redirect về home
HOMEPAGE_TITLE_SIGNATURES = [
    "tin tức 24h",
    "báo tuổi trẻ",
    "tin nhanh",
    "nhiều người xem nhất",
    "trang chủ",
]

# Headers giả lập trình duyệt thật
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


# ============================================================
# Hàm trích xuất nội dung theo từng trang báo
# ============================================================

def _extract_vnexpress(soup: BeautifulSoup) -> tuple[str, str]:
    """Trích xuất title + content từ VnExpress."""
    # Title
    title = ""
    h1 = soup.find("h1", class_="title-detail")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

    # Description (sapo)
    desc = ""
    desc_tag = soup.find("p", class_="description")
    if desc_tag:
        desc = desc_tag.get_text(strip=True)

    # Content body
    content_div = (
        soup.find("article", class_="fck_detail")
        or soup.find("div", class_="fck_detail")
    )
    paragraphs = []
    if content_div:
        for p in content_div.find_all(["p", "h2", "h3"]):
            text = p.get_text(strip=True)
            if text and len(text) > 10:
                paragraphs.append(text)

    if desc:
        paragraphs.insert(0, desc)

    return title, "\n\n".join(paragraphs)


def _extract_tuoitre(soup: BeautifulSoup) -> tuple[str, str]:
    """Trích xuất title + content từ Tuổi Trẻ."""
    # Title
    title = ""
    h1 = soup.find("h1", class_="detail-title")
    if not h1:
        h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # Sapo
    desc = ""
    sapo = soup.find("h2", class_="detail-sapo")
    if sapo:
        desc = sapo.get_text(strip=True)

    # Content
    content_div = (
        soup.find("div", class_="detail-content")
        or soup.find("div", id="main-detail-body")
        or soup.find("div", class_="content-detail")
    )
    paragraphs = []
    if content_div:
        for p in content_div.find_all(["p", "h2", "h3"]):
            # Bỏ qua các class không phải nội dung
            cls = " ".join(p.get("class", []))
            if "VCObjectBoxRelatedNewsItemTitle" in cls:
                continue
            text = p.get_text(strip=True)
            if text and len(text) > 10:
                paragraphs.append(text)

    if desc:
        paragraphs.insert(0, desc)

    return title, "\n\n".join(paragraphs)


def _extract_thanhnien(soup: BeautifulSoup) -> tuple[str, str]:
    """Trích xuất title + content từ Thanh Niên."""
    # Title
    title = ""
    h1 = soup.find("h1", class_="detail-title")
    if not h1:
        h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # Sapo
    desc = ""
    sapo = soup.find("div", class_="detail-sapo") or soup.find("h2", class_="detail-sapo")
    if sapo:
        desc = sapo.get_text(strip=True)

    # Content
    content_div = (
        soup.find("div", class_="detail-content")
        or soup.find("div", class_="detail__content")
        or soup.find("div", id="abody")
        or soup.find("div", class_="article-content")
    )
    paragraphs = []
    if content_div:
        for p in content_div.find_all(["p", "h2", "h3"]):
            text = p.get_text(strip=True)
            if text and len(text) > 10:
                paragraphs.append(text)

    if desc:
        paragraphs.insert(0, desc)

    return title, "\n\n".join(paragraphs)


def _extract_dantri(soup: BeautifulSoup) -> tuple[str, str]:
    """Trích xuất title + content từ Dân Trí."""
    # Title
    title = ""
    h1 = soup.find("h1", class_="title-page")
    if not h1:
        h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)

    # Sapo
    desc = ""
    sapo = soup.find("h2", class_="singular-sapo")
    if sapo:
        desc = sapo.get_text(strip=True)

    # Content
    content_div = (
        soup.find("div", class_="singular-content")
        or soup.find("div", class_="content-detail")
    )
    paragraphs = []
    if content_div:
        for p in content_div.find_all(["p", "h2", "h3"]):
            text = p.get_text(strip=True)
            if text and len(text) > 10:
                paragraphs.append(text)

    if desc:
        paragraphs.insert(0, desc)

    return title, "\n\n".join(paragraphs)


def _extract_generic(soup: BeautifulSoup) -> tuple[str, str]:
    """Fallback: trích xuất nội dung bài báo bằng heuristic chung."""
    # Title
    title = ""
    h1 = soup.find("h1")
    if h1:
        title = h1.get_text(strip=True)
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

    # Xóa noise
    for tag in soup(["script", "style", "noscript", "nav", "header",
                      "footer", "aside", "iframe", "form"]):
        tag.decompose()

    # Thử tìm container phổ biến
    selectors = [
        {"class_": "fck_detail"},
        {"class_": "detail-content"},
        {"class_": "article-content"},
        {"class_": "content-detail"},
        {"class_": "singular-content"},
        {"class_": "the-article-body"},
        {"id": "abody"},
    ]
    content_div = None
    for sel in selectors:
        content_div = soup.find("article", **sel) or soup.find("div", **sel)
        if content_div:
            break

    if not content_div:
        content_div = soup.find("body") or soup

    paragraphs = []
    for p in content_div.find_all(["p", "h2", "h3"]):
        text = p.get_text(strip=True)
        if text and len(text) > 10:
            paragraphs.append(text)

    # Nếu quá ít paragraph, fallback lấy raw text
    if len(paragraphs) < 3:
        raw = content_div.get_text("\n", strip=True)
        paragraphs = [
            line.strip() for line in raw.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]

    return title, "\n\n".join(paragraphs)


def _extract_jsonld(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Trích xuất title + content từ JSON-LD (schema.org NewsArticle).

    Đây là phương pháp ỔN ĐỊNH NHẤT, độc lập với layout từng trang báo:
    hầu hết báo lớn (VnExpress, Tuổi Trẻ, Thanh Niên, Dân Trí) đều nhúng
    <script type="application/ld+json"> chứa 'headline' và 'articleBody'.
    """
    title, content = "", ""

    for script in soup.find_all("script", type="application/ld+json"):
        raw = script.string or script.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue

        # JSON-LD có thể là 1 object, 1 list, hoặc có @graph
        candidates = []
        if isinstance(data, list):
            candidates = data
        elif isinstance(data, dict):
            candidates = data.get("@graph", [data])

        for obj in candidates:
            if not isinstance(obj, dict):
                continue
            obj_type = obj.get("@type", "")
            types = obj_type if isinstance(obj_type, list) else [obj_type]
            if any("Article" in str(t) for t in types):
                body = obj.get("articleBody", "")
                if body and len(body) > len(content):
                    content = body
                    title = obj.get("headline", "") or title

    return title, content


def _detect_site(url: str) -> str:
    """Nhận diện trang báo từ URL."""
    if "vnexpress.net" in url:
        return "vnexpress"
    if "tuoitre.vn" in url:
        return "tuoitre"
    if "thanhnien.vn" in url:
        return "thanhnien"
    if "dantri.com.vn" in url:
        return "dantri"
    return "generic"


# ============================================================
# Hàm crawl chính
# ============================================================

def _is_valid_article(article: dict) -> tuple[bool, str]:
    """
    Kiểm tra bài crawl có hợp lệ không (chống lưu rác/trang chủ).

    Returns:
        (is_valid, reason) — reason mô tả lý do nếu không hợp lệ.
    """
    content = article.get("content_markdown", "")
    title = (article.get("title") or "").lower()

    if len(content) < MIN_CONTENT_CHARS:
        return False, f"nội dung quá ngắn ({len(content)} < {MIN_CONTENT_CHARS} ký tự)"

    # Nếu title trùng tiêu đề trang chủ → URL đã redirect về home
    for sig in HOMEPAGE_TITLE_SIGNATURES:
        if sig in title:
            return False, f"title giống trang chủ ('{sig}') — URL có thể đã chết/redirect"

    return True, "ok"


def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo bằng requests + BeautifulSoup.

    Thứ tự trích xuất (ưu tiên cái ổn định nhất):
        1. JSON-LD articleBody  (độc lập layout, ổn định nhất)
        2. Site-specific extractor (selector riêng từng báo)
        3. Generic heuristic fallback
    Tự chọn kết quả có nội dung dài nhất.
    """
    res = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
    res.raise_for_status()

    # Detect encoding
    if res.encoding and "iso-8859" in res.encoding.lower():
        res.encoding = "utf-8"
    if not res.encoding:
        res.encoding = "utf-8"

    # Cảnh báo nếu bị redirect (dấu hiệu URL gốc đã chết)
    final_url = res.url
    if final_url.rstrip("/") != url.rstrip("/"):
        print(f"    ⚠ Redirect: {url}\n              → {final_url}")

    soup = BeautifulSoup(res.text, "html.parser")
    site = _detect_site(url)

    extractors = {
        "vnexpress": _extract_vnexpress,
        "tuoitre": _extract_tuoitre,
        "thanhnien": _extract_thanhnien,
        "dantri": _extract_dantri,
    }

    # 1) JSON-LD trước
    title, content = _extract_jsonld(soup)

    # 2) Site-specific — dùng nếu cho nội dung dài hơn
    site_extractor = extractors.get(site, _extract_generic)
    title_site, content_site = site_extractor(soup)
    if len(content_site) > len(content):
        content = content_site
    if not title and title_site:
        title = title_site

    # 3) Generic fallback — chỉ dùng khi 2 cách trên vẫn quá ít
    if len(content) < MIN_CONTENT_CHARS:
        title_fb, content_fb = _extract_generic(soup)
        if len(content_fb) > len(content):
            content = content_fb
        if not title and title_fb:
            title = title_fb

    # Clean
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+", " ", content).strip()

    if not title:
        title = "Unknown"

    return {
        "url": url,
        "final_url": final_url,
        "title": title,
        "source": site,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content,
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    print("=" * 60)
    print("Task 2 — Crawl bài báo nghệ sĩ VN liên quan ma túy")
    print(f"  Số bài: {len(ARTICLE_URLS)}")
    print(f"  Thư mục: {DATA_DIR}")
    print("=" * 60)

    # Dọn các file article_*.json cũ (tránh lẫn rác từ lần chạy trước)
    for old in DATA_DIR.glob("article_*.json"):
        old.unlink()

    saved_count = 0
    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"\n[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

        try:
            article = crawl_article(url)

            # Kiểm tra chất lượng trước khi lưu — không lưu rác/trang chủ
            is_valid, reason = _is_valid_article(article)
            if not is_valid:
                print(f"  ❌ BỎ QUA (không hợp lệ): {reason}")
                print(f"     Title: {article['title']}")
                continue

            # Lưu file JSON
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            saved_count += 1
            print(f"  ✅ Saved: {filepath.name}")
            print(f"     Title: {article['title']}")
            print(f"     Content: {len(article['content_markdown']):,} chars")

        except Exception as e:
            print(f"  ❌ Lỗi: {e}")

        # Rate limiting: chờ giữa các request
        if i < len(ARTICLE_URLS):
            time.sleep(2)

    # Tổng kết
    print("\n" + "=" * 60)
    print("Tổng kết:")
    json_files = sorted(DATA_DIR.glob("*.json"))
    for f in json_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        content_len = len(data.get("content_markdown", ""))
        print(f"  📰 {f.name} — {data.get('title', 'N/A')[:60]} ({content_len:,} chars)")
    print(f"  Tổng: {len(json_files)} file(s) hợp lệ đã lưu")
    if len(json_files) < MIN_ARTICLES_REQUIRED:
        print(f"  ⚠ CẢNH BÁO: cần tối thiểu {MIN_ARTICLES_REQUIRED} bài, "
              f"hiện chỉ có {len(json_files)}. Hãy bổ sung URL vào ARTICLE_URLS.")
    else:
        print(f"  ✅ Đạt yêu cầu (≥{MIN_ARTICLES_REQUIRED} bài).")
    print("=" * 60)


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
    else:
        crawl_all()
