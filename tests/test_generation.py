from langchain_core.documents import Document

from ragcourse.generation import PROMPT, format_context


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
