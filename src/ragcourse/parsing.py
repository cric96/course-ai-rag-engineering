"""Turn documents into plain text, one string per page."""

import re
from pathlib import Path

import pymupdf


def extract_pages(path: Path) -> list[str]:
    """Return the text layer of each PDF page ("" when the page has none)."""
    with pymupdf.open(path) as pdf:
        return [clean_text(page.get_text()) for page in pdf]


def clean_text(text: str) -> str:
    """Normalize whitespace and rejoin words split across lines."""
    # "rendi-\ncontazione" -> "rendicontazione" (lowercase only: keeps "2021-\n2027").
    text = re.sub(r"([a-zà-ù])-\n([a-zà-ù])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)  # runs of spaces -> one space
    text = re.sub(r" *\n *", "\n", text)  # no spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)  # at most one blank line
    return text.strip()


def render_page(path: Path, page: int, dpi: int = 100) -> bytes:
    """Render one PDF page (1-based) as PNG bytes, e.g. to show it to a vision model."""
    with pymupdf.open(path) as pdf:
        return pdf[page - 1].get_pixmap(dpi=dpi).tobytes("png")
