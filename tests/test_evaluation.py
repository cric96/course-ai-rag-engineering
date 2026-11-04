from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ragcourse.evaluation import gold_found, gold_qrels, retrieval_run
from ragcourse.filters import metadata_filter

QUESTIONS = [
    {
        "id": "q1",
        "question": "Chi?",
        "call": "bando-a",
        "gold": [{"url": "u", "page": 2}],
    }
]


def chunk(page: int) -> Document:
    return Document("testo", metadata={"url": "u", "page": page})


class FakeRetriever(BaseRetriever):
    """Always returns chunks of pages 1, 2, 2, 3."""

    def _get_relevant_documents(self, query, *, run_manager):
        return [chunk(1), chunk(2), chunk(2), chunk(3)]


def test_gold_qrels_marks_the_gold_pages_as_relevant():
    assert gold_qrels(QUESTIONS).to_dict() == {"q1": {"u#page=2": 1}}


def test_retrieval_run_ranks_pages_by_their_first_chunk():
    run = retrieval_run(FakeRetriever(), QUESTIONS, name="fake")

    assert run.name == "fake"
    assert run.to_dict() == {"q1": {"u#page=1": 3, "u#page=2": 2, "u#page=3": 1}}


class RecordingRetriever(BaseRetriever):
    """Remembers the search options of every query."""

    options: list[dict] = []

    def _get_relevant_documents(self, query, *, run_manager, **options):
        self.options.append(options)
        return [chunk(2)]


def test_retrieval_run_searches_everything_by_default():
    retriever = RecordingRetriever()
    retrieval_run(retriever, QUESTIONS, name="tutto")
    assert retriever.options == [{}]


def test_retrieval_run_can_search_only_the_call_of_each_question():
    retriever = RecordingRetriever()
    retrieval_run(retriever, QUESTIONS, name="per bando", by_call=True)
    assert retriever.options == [{"filter": metadata_filter(call="bando-a")}]


def test_gold_found_tells_whether_any_chunk_comes_from_a_gold_page():
    assert gold_found([chunk(1), chunk(2)], QUESTIONS[0])
    assert not gold_found([chunk(1), chunk(3)], QUESTIONS[0])
