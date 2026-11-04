import base64
from pathlib import Path

import pymupdf
import pytest
from langchain_core.messages import HumanMessage

from ragcourse.evalset import (
    FaqLabel,
    ImageQuestion,
    ask_image_questions,
    ask_labels,
    dedupe_faqs,
    image_question_message,
    label_pages,
    labeling_message,
    page_parts,
    to_image_questions,
    to_questions,
    visual_pages,
)
from ragcourse.models import LABELING_MODEL, chat_model


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
    assert part["type"] == "text"
    assert part["text"].startswith("[P7] Bando, p. 3")
    assert text in part["text"]


def test_page_parts_sends_an_image_when_text_is_missing(tmp_path: Path):
    label, image = page_parts("P7", page("", path=blank_pdf(tmp_path)))
    assert label["text"].startswith("[P7]")
    assert (image["type"], image["mime_type"]) == ("image", "image/png")
    assert base64.b64decode(image["base64"]).startswith(b"\x89PNG")


def test_labeling_message_ends_with_numbered_faq():
    message = labeling_message([faq("Chi?"), faq("Quando?")], {})
    assert isinstance(message, HumanMessage)
    assert "[1] Q: Chi?" in message.content[-1]["text"]
    assert "[2] Q: Quando?" in message.content[-1]["text"]


def test_to_questions_maps_labels_to_gold_pages_and_ignores_unknown_labels():
    labeled = label_pages([page("a", number=4)])
    labels = [FaqLabel(faq=1, pages=["P1", "P99"], evidence="text")]

    [question] = to_questions([faq("Chi?")], labels, labeled)

    assert question["gold"] == [{"url": "https://fesr/bando.pdf", "page": 4}]
    assert question["evidence"] == "text"


def test_to_questions_marks_faq_without_label_as_unsupported():
    [question] = to_questions([faq("Chi?")], [], {})
    assert (question["gold"], question["evidence"]) == ([], "none")


def blank_pdf(tmp_path: Path, text: str = "") -> str:
    path = tmp_path / "slide.pdf"
    with pymupdf.open() as doc:
        doc.new_page().insert_text((72, 144), text, fontsize=28)
        doc.save(path)
    return str(path)


def test_visual_pages_keeps_pages_with_little_text():
    pages = [page("SPESE AMMISSIBILI"), page("Testo lungo " * 10)]
    assert visual_pages(pages) == [pages[0]]


def test_image_question_contents_shows_every_page_as_image(tmp_path: Path):
    labeled = label_pages([page("Titolo", path=blank_pdf(tmp_path))])
    message = image_question_message(labeled, n_questions=5)

    assert any(part["type"] == "image" for part in message.content)
    assert "5" in message.content[0]["text"]  # the prompt asks for that many


def test_to_image_questions_use_the_source_page_as_gold():
    labeled = label_pages([page("", number=14)])
    generated = [ImageQuestion(page="P1", question="Entro quando?", answer="30/9/2027")]

    [question] = to_image_questions(generated, labeled)

    assert question["gold"] == [{"url": "https://fesr/bando.pdf", "page": 14}]
    assert (question["source"], question["evidence"]) == ("image", "image")


DEADLINE = "Scadenza: 30/9/2027"  # short: the model receives this page as an image


@pytest.mark.llm  # one small request to the labeling model
def test_labeling_model_finds_the_answer_in_an_image_page(tmp_path: Path):
    image_page = page(DEADLINE, path=blank_pdf(tmp_path, DEADLINE))
    pages = label_pages([page("Testo lungo " * 10), image_page])

    [label] = ask_labels(chat_model(LABELING_MODEL), [faq("Entro quando?")], pages)

    assert (label.pages, label.evidence) == (["P2"], "image")


@pytest.mark.llm  # one small request to the labeling model
def test_labeling_model_writes_questions_from_an_image_page(tmp_path: Path):
    pages = label_pages([page(DEADLINE, path=blank_pdf(tmp_path, DEADLINE))])

    [generated] = ask_image_questions(chat_model(LABELING_MODEL), pages, n_questions=1)

    assert generated.page == "P1"
    assert "2027" in generated.answer
