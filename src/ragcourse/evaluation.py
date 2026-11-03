"""Score a retriever on the evaluation set."""

from collections.abc import Callable

import pandas as pd

from ragcourse.metrics import recall_at_k, reciprocal_rank
from ragcourse.retrieval import Hit

Search = Callable[[str, int], list[Hit]]  # (question, k) -> best k hits


def evaluate_retrieval(
    search: Search, questions: list[dict], ks: tuple[int, ...] = (1, 3, 5, 10)
) -> pd.DataFrame:
    """One row per question: recall@k for each k and reciprocal rank (mean = MRR)."""
    rows = []
    for question in questions:
        gold = {(page["url"], page["page"]) for page in question["gold"]}
        if not gold:
            continue  # nothing to find: these questions test abstention instead
        hits = search(question["question"], max(ks))
        pages = [(hit.chunk.url, hit.chunk.page) for hit in hits]

        row = {"id": question["id"], "evidence": question["evidence"]}
        row |= {f"recall@{k}": recall_at_k(pages, gold, k) for k in ks}
        row["mrr"] = reciprocal_rank(pages, gold)
        rows.append(row)
    return pd.DataFrame(rows)
