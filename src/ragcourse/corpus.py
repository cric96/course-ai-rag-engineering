"""The corpus as LangChain documents: one Document per page with text."""

from pathlib import Path

from langchain_core.documents import Document

from ragcourse.io import read_jsonl
from ragcourse.paths import DATA

PAGES = DATA / "processed/pages.jsonl"  # built by scripts/extract_pages.py


def load_pages(path: Path = PAGES) -> list[Document]:
    """Pages with text, each carrying where it comes from (url, page, title, call)."""
    return [
        Document(
            page_content=page["text"],
            metadata={key: page[key] for key in ("url", "page", "title", "call")},
        )
        for page in read_jsonl(path)
        if page["text"]  # pages without a text layer are invisible to text search
    ]


def with_headers(docs: list[Document]) -> list[Document]:
    """Prepend document title and page to each document's text."""
    return [
        Document(
            page_content=f"[{doc.metadata['title']}, p. {doc.metadata['page']}]\n"
            f"{doc.page_content}",
            metadata=doc.metadata,
            id=doc.id,
        )
        for doc in docs
    ]
