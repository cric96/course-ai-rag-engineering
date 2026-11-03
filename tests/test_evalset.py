from pathlib import Path

import pymupdf
from google.genai import types

from ragcourse.evalset import (
    FaqLabel,
    dedupe_faqs,
    label_pages,
    labeling_contents,
    page_parts,
    to_questions,
)


def faq(question: str, call: str = "bando-a", answer: str = "Sì.") -> dict:
    return {"question": question, "answer": answer, "call": call, "url": "u"}


def page(text: str, number: int = 1, path: str = "x.pdf", kind: str = "file") -> dict:
    return {
        "url": "https://fesr/bando.pdf",
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
    path = tmp_path / "slide.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path)

    label, image = page_parts("P7", page("", path=str(path)))
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
