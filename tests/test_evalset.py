from pathlib import Path

import pymupdf
from google.genai import types

from ragcourse.evalset import (
    FaqLabel,
    ImageQuestion,
    dedupe_faqs,
    image_question_contents,
    label_pages,
    labeling_contents,
    page_parts,
    to_image_questions,
    to_questions,
    visual_pages,
)


def faq(question: str, call: str = "bando-a", answer: str = "Sì.") -> dict:
    return {"question": question, "answer": answer, "call": call, "url": "u"}


def page(text: str, number: int = 1, path: str = "x.pdf", kind: str = "file") -> dict:
    return {
        "url": "https://fesr/bando.pdf",
        "call": "bando-a",
        "title": "Bando",
        "page": number,
        "text": text,
        "path": path,
        "kind": kind,
    }


def test_dedupe_drops_repeated_questions_of_the_same_call():
    faqs = [faq("Chi può partecipare?"), faq("chi può partecipare?"), faq("Altro?")]
    assert [f["question"] for f in dedupe_faqs(faqs)] == [
        "Chi può partecipare?",
        "Altro?",
    ]


def test_dedupe_keeps_same_question_in_different_calls():
    faqs = [faq("Chi può partecipare?", "bando-a"), faq("Chi può partecipare?", "b")]
    assert len(dedupe_faqs(faqs)) == 2


def test_label_pages_numbers_pages_from_p1():
    labeled = label_pages([page("a"), page("b", number=2)])
    assert list(labeled) == ["P1", "P2"]


def test_page_parts_sends_text_when_the_page_has_enough():
    text = "Le spese ammissibili sono quelle sostenute dopo la domanda."
    [part] = page_parts("P7", page(text, number=3))
    assert part.startswith("[P7] Bando, p. 3")
    assert text in part


def test_page_parts_sends_an_image_when_text_is_missing(tmp_path: Path):
    label, image = page_parts("P7", page("", path=blank_pdf(tmp_path)))
    assert label.startswith("[P7]")
    assert isinstance(image, types.Part)
    assert image.inline_data.mime_type == "image/png"


def test_labeling_contents_ends_with_numbered_faq():
    contents = labeling_contents([faq("Chi?"), faq("Quando?")], {})
    assert "[1] Q: Chi?" in contents[-1]
    assert "[2] Q: Quando?" in contents[-1]


def test_to_questions_maps_labels_to_gold_pages_and_ignores_unknown_labels():
    labeled = label_pages([page("a", number=4)])
    labels = [FaqLabel(faq=1, pages=["P1", "P99"], evidence="text")]

    [question] = to_questions([faq("Chi?")], labels, labeled)

    assert question["gold"] == [{"url": "https://fesr/bando.pdf", "page": 4}]
    assert question["evidence"] == "text"


def test_to_questions_marks_faq_without_label_as_unsupported():
    [question] = to_questions([faq("Chi?")], [], {})
    assert (question["gold"], question["evidence"]) == ([], "none")


def blank_pdf(tmp_path: Path) -> str:
    path = tmp_path / "slide.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path)
    return str(path)


def test_visual_pages_keeps_pages_with_little_text():
    pages = [page("SPESE AMMISSIBILI"), page("Testo lungo " * 10)]
    assert visual_pages(pages) == [pages[0]]


def test_image_question_contents_shows_every_page_as_image(tmp_path: Path):
    labeled = label_pages([page("Titolo", path=blank_pdf(tmp_path))])
    contents = image_question_contents(labeled, n_questions=5)

    assert any(isinstance(part, types.Part) for part in contents)
    assert "5" in contents[0]  # the prompt asks for the requested number


def test_to_image_questions_use_the_source_page_as_gold():
    labeled = label_pages([page("", number=14)])
    generated = [ImageQuestion(page="P1", question="Entro quando?", answer="30/9/2027")]

    [question] = to_image_questions(generated, labeled)

    assert question["gold"] == [{"url": "https://fesr/bando.pdf", "page": 14}]
    assert (question["source"], question["evidence"]) == ("image", "image")
