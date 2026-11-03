from ragcourse.chunking import Chunk
from ragcourse.evaluation import evaluate_retrieval
from ragcourse.retrieval import Hit

CHUNKS = [Chunk("testo", "bando.pdf", page, "Bando") for page in (1, 2, 3)]


def fake_search(question: str, k: int) -> list[Hit]:
    return [Hit(c, 1.0) for c in CHUNKS[:k]]  # always pages 1, 2, 3


def question(gold_pages: list[int]) -> dict:
    gold = [{"url": "bando.pdf", "page": p} for p in gold_pages]
    return {"id": "q", "question": "?", "evidence": "text", "gold": gold}


def test_evaluate_retrieval_scores_each_question_on_its_gold_pages():
    results = evaluate_retrieval(fake_search, [question([2])], ks=(1, 3))
    [row] = results.to_dict("records")
    assert (row["hit@1"], row["hit@3"]) == (0.0, 1.0)
    assert (row["recall@1"], row["recall@3"], row["mrr"]) == (0.0, 1.0, 0.5)


def test_evaluate_retrieval_keeps_the_question_text_for_error_analysis():
    results = evaluate_retrieval(fake_search, [question([2])])
    assert results.loc[0, "question"] == "?"


def test_evaluate_retrieval_skips_questions_without_gold_pages():
    results = evaluate_retrieval(fake_search, [question([]), question([1])])
    assert len(results) == 1
