"""Glue for the course chat: readable call names and citations that open the PDF."""

import re

from langchain_core.documents import Document

from ragcourse.evaluation import page_id

CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [2] or [1, 3]


def call_label(slug: str) -> str:
    """The call's slug as a name: '2019' between letters encodes an apostrophe."""
    text = re.sub(r"(?<=[a-z])2019(?=[a-z])", "’", slug).replace("-", " ")
    return text[:1].upper() + text[1:]


def numbers(match: re.Match) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", match.group(1))]


def cited(answer: str, chunks: list[Document]) -> list[int]:
    """The chunk numbers the answer cites, each once, in order of first citation."""
    found = [n for m in CITATION.finditer(answer) for n in numbers(m)]
    return [n for n in dict.fromkeys(found) if 1 <= n <= len(chunks)]


def link_citations(answer: str, chunks: list[Document]) -> str:
    """Markdown where each [n] links to the page of the PDF it comes from."""

    def link(n: int) -> str:
        if not 1 <= n <= len(chunks):
            return str(n)
        meta = chunks[n - 1].metadata
        return f"[{n}]({page_id(meta['url'], meta['page'])})"

    def replace(match: re.Match) -> str:
        return r"\[" + ", ".join(link(n) for n in numbers(match)) + r"\]"

    return CITATION.sub(replace, answer)
