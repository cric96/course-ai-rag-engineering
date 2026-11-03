"""Evaluate retrieval with ranx, the standard library for search metrics."""

from langchain_core.retrievers import BaseRetriever
from ranx import Qrels, Run


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


def retrieval_run(retriever: BaseRetriever, questions: list[dict], name: str) -> Run:
    """Ask the retriever every question; rank pages by their first retrieved chunk."""
    results = retriever.batch([q["question"] for q in questions])  # runs in parallel
    run = {}
    for question, chunks in zip(questions, results, strict=True):
        # Several chunks may come from the same page: keep its best position.
        pages = dict.fromkeys(
            page_id(c.metadata["url"], c.metadata["page"]) for c in chunks
        )
        run[question["id"]] = {p: len(pages) - rank for rank, p in enumerate(pages)}
    return Run(run, name=name)
