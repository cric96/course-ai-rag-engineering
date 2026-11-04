import pymupdf
from langchain_core.documents import Document

from ragcourse.generation import (
    ABSTAIN,
    PROMPT,
    PRUDENT_PROMPT,
    format_context,
    picture_messages,
)


def test_format_context_numbers_chunks_so_the_answer_can_cite_them():
    chunks = [
        Document(
            "Le microimprese sono ammesse.", metadata={"title": "Bando", "page": 4}
        ),
        Document("Scadenza: 30/9/2027.", metadata={"title": "FAQ", "page": 1}),
    ]
    context = format_context(chunks)

    assert context.startswith("[1] Bando, p. 4\nLe microimprese sono ammesse.")
    assert "[2] FAQ, p. 1\nScadenza: 30/9/2027." in context


def test_prompt_puts_context_and_question_in_the_message():
    messages = PROMPT.invoke({"context": "[1] Bando", "question": "Chi?"}).to_messages()

    assert "solo" in messages[0].text  # answer only from the documents
    assert "[1] Bando" in messages[-1].text
    assert "Chi?" in messages[-1].text


def test_prompt_asks_for_a_fixed_sentence_when_the_answer_is_missing():
    # A fixed sentence makes abstentions countable: answer.startswith(ABSTAIN).
    system = PROMPT.invoke({"context": "", "question": ""}).to_messages()[0].text
    assert ABSTAIN in system


def system_text(prompt) -> str:
    return prompt.invoke({"context": "", "question": ""}).to_messages()[0].text


def test_prudent_prompt_differs_only_in_when_to_abstain():
    default, prudent = system_text(PROMPT), system_text(PRUDENT_PROMPT)

    assert "solo in parte" in default
    assert "solo in parte" not in prudent
    assert ABSTAIN in prudent
    assert default.splitlines()[:4] == prudent.splitlines()[:4]  # same other rules


def test_picture_messages_show_numbered_page_images_then_the_question(tmp_path):
    path = tmp_path / "slide.pdf"
    with pymupdf.open() as doc:
        doc.new_page()
        doc.save(path)
    page = {"path": str(path), "page": 1, "kind": "file", "title": "Slide"}

    system, human = picture_messages("Chi?", [page, page])

    assert ABSTAIN in system.text  # same rules as the text prompt
    kinds = [block["type"] for block in human.content]
    assert kinds == ["text", "text", "image", "text", "image", "text"]
    assert human.content[1]["text"] == "[1] Slide, p. 1"
    assert human.content[-1]["text"].endswith("Domanda: Chi?")
