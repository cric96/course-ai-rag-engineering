"""Generation: the prompt that turns retrieved chunks into a cited answer."""

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from ragcourse.evalset import page_image

ABSTAIN = "Non lo trovo nei documenti."  # fixed, so abstentions can be counted

RULES = (
    "Rispondi a domande sui bandi del PR FESR 2021-2027 dell'Emilia-Romagna.\n"
    "Usa solo i documenti forniti. "
    "Rispondi in modo preciso, con le condizioni\n"
    "e le eccezioni che contano.\n"
    "Cita le fonti con il loro numero tra parentesi quadre, ad esempio [2].\n"
)
QUESTION = ("human", "Documenti:\n\n{context}\n\nDomanda: {question}")

# When to abstain: the one rule that differs between the two prompts.
PARTIAL_OK = (
    "Se i documenti rispondono solo in parte, rispondi con quello che dicono.\n"
    f"Se non contengono nulla di utile, rispondi solo: {ABSTAIN}"
)
COMPLETE_ONLY = (
    f"Se i documenti non contengono la risposta completa, rispondi solo: {ABSTAIN}"
)

PROMPT = ChatPromptTemplate.from_messages([("system", RULES + PARTIAL_OK), QUESTION])
PRUDENT_PROMPT = ChatPromptTemplate.from_messages(
    [("system", RULES + COMPLETE_ONLY), QUESTION]
)


def format_context(chunks: list[Document]) -> str:
    """Numbered chunks with their source, so the answer can cite them as [1], [2]."""
    return "\n\n".join(
        f"[{n}] {chunk.metadata['title']}, p. {chunk.metadata['page']}\n"
        f"{chunk.page_content}"
        for n, chunk in enumerate(chunks, start=1)
    )


def expand_pages(
    chunks: list[Document], pages: dict[tuple[str, int], Document]
) -> list[Document]:
    """Replace selected chunks with full pages, deduplicating in retrieval order."""
    selected = dict.fromkeys((c.metadata["url"], c.metadata["page"]) for c in chunks)
    return [pages[key] for key in selected]


def picture_messages(question: str, pages: list[dict]) -> list[BaseMessage]:
    """The default prompt, with page images instead of text chunks (visual RAG)."""
    blocks = [{"type": "text", "text": "Documenti:"}]
    for n, page in enumerate(pages, start=1):
        blocks += [
            {"type": "text", "text": f"[{n}] {page['title']}, p. {page['page']}"},
            page_image(page),
        ]
    blocks.append({"type": "text", "text": f"Domanda: {question}"})
    return [SystemMessage(RULES + PARTIAL_OK), HumanMessage(content=blocks)]
