"""Retrieval metrics: did the search put the right pages near the top?"""

from collections.abc import Hashable


def recall_at_k(retrieved: list[Hashable], relevant: set, k: int) -> float:
    """Share of the relevant items that appear in the first k results."""
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[Hashable], relevant: set) -> float:
    """1 / position of the first relevant result, 0 if there is none."""
    for position, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1 / position
    return 0.0
