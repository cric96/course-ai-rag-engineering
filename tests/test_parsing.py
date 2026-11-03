from pathlib import Path

import pymupdf
import pytest

from ragcourse.parsing import clean_text, extract_pages, render_page


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    # Two pages: one with text, one blank (like a scanned or image-only page).
    path = tmp_path / "bando.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 72), "Scadenza: 30 settembre 2027")
        doc.new_page()
        doc.save(path)
    return path


def test_extract_pages_returns_one_text_per_page(pdf: Path):
    assert extract_pages(pdf) == ["Scadenza: 30 settembre 2027", ""]


def test_clean_text_collapses_spaces_and_blank_lines():
    assert clean_text("Bando   FESR\n\n\n\n2026 ") == "Bando FESR\n\n2026"


def test_clean_text_joins_words_hyphenated_at_line_end():
    assert clean_text("rendi-\ncontazione") == "rendicontazione"


def test_clean_text_keeps_hyphens_inside_lines():
    assert clean_text("PR FESR 2021-2027") == "PR FESR 2021-2027"


def test_render_page_returns_png_of_the_requested_page(pdf: Path):
    png = render_page(pdf, page=2, dpi=50)
    assert png.startswith(b"\x89PNG")
