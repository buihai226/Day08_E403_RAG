"""
Tiện ích: sinh file PDF từ các văn bản luật .txt trong data/landing/legal/.

Mục đích: PageIndex (Task 8) chỉ nhận PDF. Ta đã có nội dung luật dạng .txt,
nên convert sang .pdf để upload lên PageIndex.

Dùng reportlab + font Unicode của Windows (Arial/Times) để hiển thị đúng
tiếng Việt có dấu.

Chạy:
    python src/make_legal_pdfs.py
"""

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Các font Unicode thường có sẵn trên Windows (hỗ trợ tiếng Việt)
FONT_CANDIDATES = [
    ("VNFont", r"C:\Windows\Fonts\arial.ttf"),
    ("VNFont", r"C:\Windows\Fonts\times.ttf"),
    ("VNFont", r"C:\Windows\Fonts\segoeui.ttf"),
]


def _register_font() -> str:
    """Đăng ký font Unicode đầu tiên tìm thấy. Trả tên font dùng được."""
    for name, path in FONT_CANDIDATES:
        if Path(path).exists():
            pdfmetrics.registerFont(TTFont(name, path))
            print(f"  Dùng font: {path}")
            return name
    print("  ⚠ Không tìm thấy font Unicode — dùng Helvetica (có thể lỗi dấu tiếng Việt).")
    return "Helvetica"


def txt_to_pdf(txt_path: Path, font_name: str):
    """Convert 1 file .txt → .pdf cùng tên (cùng thư mục)."""
    pdf_path = txt_path.with_suffix(".pdf")
    text = txt_path.read_text(encoding="utf-8")

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "VNBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=11,
        leading=16,
    )

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
    )

    flowables = []
    # Tách theo dòng; dòng trống → khoảng cách
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            flowables.append(Spacer(1, 6))
            continue
        flowables.append(Paragraph(escape(line), body))

    doc.build(flowables)
    print(f"  ✅ {txt_path.name} → {pdf_path.name} ({pdf_path.stat().st_size:,} bytes)")


def main():
    print("=" * 60)
    print("Sinh PDF từ văn bản luật .txt cho PageIndex (Task 8)")
    print("=" * 60)

    font_name = _register_font()

    txt_files = sorted(LEGAL_DIR.glob("*.txt"))
    if not txt_files:
        print(f"  ⚠ Không có file .txt trong {LEGAL_DIR}")
        return

    for txt in txt_files:
        txt_to_pdf(txt, font_name)

    print(f"\n  Tổng: {len(txt_files)} PDF đã tạo trong {LEGAL_DIR}")


if __name__ == "__main__":
    main()
