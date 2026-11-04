"""Generation: the prompt that turns retrieved chunks into a cited answer."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rispondi a domande sui bandi del PR FESR 2021-2027 dell'Emilia-Romagna.\n"
            "Usa solo i documenti forniti e rispondi in modo breve e preciso.\n"
            "Cita le fonti con il loro numero tra parentesi quadre, ad esempio [2].\n"
            "Se i documenti non contengono la risposta, dillo chiaramente.",
        ),
        ("human", "Documenti:\n\n{context}\n\nDomanda: {question}"),
    ]
)


def format_context(chunks: list[Document]) -> str:
    """Numbered chunks with their source, so the answer can cite them as [1], [2]."""
    return "\n\n".join(
        f"[{n}] {chunk.metadata['title']}, p. {chunk.metadata['page']}\n"
        f"{chunk.page_content}"
        for n, chunk in enumerate(chunks, start=1)
    )
