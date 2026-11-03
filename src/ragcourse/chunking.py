"""Split page text into chunks: the pieces we embed and retrieve."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A piece of text and the page it comes from."""

    text: str
    url: str
    page: int
    title: str


def split_words(text: str, size: int, overlap: int) -> list[str]:
    """Windows of `size` words; consecutive windows share `overlap` words."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            return chunks
        start += size - overlap


def chunk_pages(pages: list[dict], size: int = 200, overlap: int = 50) -> list[Chunk]:
    """Chunk each page on its own, so every chunk points to exactly one page."""
    return [
        Chunk(text, page["url"], page["page"], page["title"])
        for page in pages
        for text in split_words(page["text"], size, overlap)
    ]
