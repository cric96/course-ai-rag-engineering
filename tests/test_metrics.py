import random
from statistics import mean

import pytest

from ragcourse.metrics import hit_at_k, recall_at_k, reciprocal_rank


def test_recall_at_k_is_the_share_of_relevant_items_in_the_top_k():
    retrieved = ["x", "a", "y", "b"]
    assert recall_at_k(retrieved, {"a", "b"}, k=2) == 0.5
    assert recall_at_k(retrieved, {"a", "b"}, k=4) == 1.0


def test_recall_counts_an_item_once_even_if_retrieved_twice():
    assert recall_at_k(["a", "a"], {"a", "b"}, k=2) == 0.5  # two chunks, same page


def test_hit_at_k_tells_if_any_relevant_item_is_in_the_top_k():
    assert hit_at_k(["x", "a", "y"], {"a", "b"}, k=1) == 0.0
    assert hit_at_k(["x", "a", "y"], {"a", "b"}, k=2) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_position():
    assert reciprocal_rank(["x", "y", "a", "b"], {"a", "b"}) == 1 / 3
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


@pytest.mark.slow  # ranx compiles with numba on first use
def test_metrics_match_ranx_on_random_rankings():
    from ranx import Qrels, Run, evaluate

    rng = random.Random(0)
    docs = [f"d{i}" for i in range(30)]
    rankings = {f"q{n}": rng.sample(docs, 10) for n in range(50)}
    relevant = {q: set(rng.sample(docs, rng.randint(1, 4))) for q in rankings}

    # ranx wants scores: higher score = better position.
    run = Run({q: {d: 10 - i for i, d in enumerate(r)} for q, r in rankings.items()})
    qrels = Qrels({q: dict.fromkeys(rel, 1) for q, rel in relevant.items()})
    expected = evaluate(qrels, run, ["hit_rate@5", "recall@5", "mrr"])

    ours_hit = mean(hit_at_k(rankings[q], relevant[q], 5) for q in rankings)
    ours_recall = mean(recall_at_k(rankings[q], relevant[q], 5) for q in rankings)
    ours_mrr = mean(reciprocal_rank(rankings[q], relevant[q]) for q in rankings)
    assert ours_hit == pytest.approx(expected["hit_rate@5"])
    assert ours_recall == pytest.approx(expected["recall@5"])
    assert ours_mrr == pytest.approx(expected["mrr"])
