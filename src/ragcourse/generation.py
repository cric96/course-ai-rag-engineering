"""Generation: the prompt that turns retrieved chunks into a cited answer."""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

ABSTAIN = "Non lo trovo nei documenti."  # fixed, so abstentions can be counted

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rispondi a domande sui bandi del PR FESR 2021-2027 dell'Emilia-Romagna.\n"
            "Usa solo i documenti forniti e rispondi in modo breve e preciso.\n"
            "Cita le fonti con il loro numero tra parentesi quadre, ad esempio [2].\n"
            f"Se i documenti non contengono la risposta, rispondi solo: {ABSTAIN}",
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
