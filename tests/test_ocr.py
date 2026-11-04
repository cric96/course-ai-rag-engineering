from pathlib import Path

import pymupdf
import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ragcourse.ocr import OCR_PROMPT, ocr_message, page_picture, tesseract, vlm_reader

TEXT = "Scadenza: 30 settembre 2027"


def slide(tmp_path: Path) -> dict:
    """A one-page PDF whose only content is a line of large text."""
    path = tmp_path / "slide.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 144), TEXT, fontsize=28)
        doc.save(path)
    return {"path": str(path), "page": 1, "kind": "file"}


def test_page_picture_renders_a_pdf_page_at_the_requested_resolution(tmp_path):
    small, large = (
        page_picture(slide(tmp_path), dpi=72),
        page_picture(slide(tmp_path), dpi=144),
    )
    assert large.width == 2 * small.width  # an A4 page: 595 points wide


def test_ocr_message_sends_the_instructions_and_the_page_image(tmp_path):
    message = ocr_message(slide(tmp_path))
    assert message.content[0]["text"] == OCR_PROMPT
    assert message.content[1]["type"] == "image"


def test_any_chat_model_that_sees_images_can_read_a_page(tmp_path):
    read = vlm_reader(GenericFakeChatModel(messages=iter([AIMessage(TEXT)])))
    assert read(slide(tmp_path)) == TEXT


@pytest.mark.slow  # needs the tesseract program and downloads the Italian model
def test_tesseract_reads_the_text_of_a_page_picture(tmp_path):
    assert "30 settembre 2027" in tesseract(slide(tmp_path))
