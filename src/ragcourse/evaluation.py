"""Evaluate retrieval with ranx, the standard library for search metrics."""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ranx import Qrels, Run

from ragcourse.filters import metadata_filter


def page_id(url: str, page: int) -> str:
    """One id per page; '#page=N' also opens the PDF at that page."""
    return f"{url}#page={page}"


def gold_qrels(questions: list[dict]) -> Qrels:
    """The answer key: for each question, the pages that contain the answer."""
    return Qrels(
        {
            q["id"]: {page_id(g["url"], g["page"]): 1 for g in q["gold"]}
            for q in questions
        }
    )


def gold_found(chunks: list[Document], question: dict) -> bool:
    """True when at least one chunk comes from a page that holds the answer."""
    gold = {page_id(g["url"], g["page"]) for g in question["gold"]}
    return any(page_id(c.metadata["url"], c.metadata["page"]) in gold for c in chunks)


def retrieval_run(
    retriever: BaseRetriever, questions: list[dict], name: str, by_call: bool = False
) -> Run:
    """Ask the retriever every question; rank pages by their first retrieved chunk.

    by_call=True searches only the question's call, known from where it is asked.
    """
    run = {}
    for question in questions:
        options = {"filter": metadata_filter(call=question["call"])} if by_call else {}
        # One at a time: batch() runs threads, and local GPU models then give
        # slightly different results from run to run.
        chunks = retriever.invoke(question["question"], **options)
        # Several chunks may come from the same page: keep its best position.
        pages = dict.fromkeys(
            page_id(c.metadata["url"], c.metadata["page"]) for c in chunks
        )
        run[question["id"]] = {p: len(pages) - rank for rank, p in enumerate(pages)}
    return Run(run, name=name)
