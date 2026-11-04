import pytest
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableLambda

from ragcourse.rerank import (
    CachedRetriever,
    RerankRetriever,
    cross_encoder,
    listwise_prompt,
    llm_listwise,
)

CHUNKS = [Document(text) for text in ["scadenze", "spese ammissibili", "CUP", "firma"]]


class FakeRetriever(BaseRetriever):
    """Always the same four chunks; remembers the search options it received."""

    options: list[dict] = []

    def _get_relevant_documents(self, query, *, run_manager, **options):
        self.options.append(options)
        return CHUNKS


def by_length(query, docs):
    return [len(doc.page_content) for doc in docs]  # longest text = most relevant


def test_reranker_reorders_the_candidates_and_keeps_the_best():
    reranker = RerankRetriever(base=FakeRetriever(), scorer=by_length, top_n=2)
    found = reranker.invoke("spese?")
    assert [doc.page_content for doc in found] == ["spese ammissibili", "scadenze"]


def test_reranker_passes_search_options_to_the_first_retriever():
    base = FakeRetriever()
    RerankRetriever(base=base, scorer=by_length).invoke("spese?", filter="bando-a")
    assert base.options == [{"filter": "bando-a"}]


class OrderingChat(GenericFakeChatModel):
    """An LLM that always orders the candidates 3, 1 (and forgets the others)."""

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: schema(order=[3, 1]))


def test_listwise_order_becomes_a_score_and_forgotten_chunks_go_last():
    score = llm_listwise(OrderingChat(messages=iter([])))
    scores = score("CUP?", CHUNKS)
    assert scores.index(max(scores)) == 2  # candidate 3, "CUP"
    assert scores[0] > scores[1] == scores[3]  # candidate 1 second, the rest tied


@pytest.mark.slow  # downloads and runs a real reranker
def test_cross_encoder_puts_the_relevant_chunk_first():
    score = cross_encoder("bge-reranker-v2-m3")
    docs = [
        Document("La domanda scade il 30 settembre."),
        Document("Le spese per l'acquisto di hardware sono ammissibili."),
    ]
    hardware, deadline = score("Posso comprare un computer?", docs)[::-1]
    assert hardware > deadline


def test_cached_retriever_searches_once_per_question_and_options():
    base = FakeRetriever()
    cached = CachedRetriever(base=base)
    cached.invoke("spese?", filter="a")
    cached.invoke("spese?", filter="a")
    cached.invoke("spese?", filter="b")
    assert base.options == [{"filter": "a"}, {"filter": "b"}]


def test_listwise_prompt_numbers_the_candidates_from_one():
    prompt = listwise_prompt("CUP?", CHUNKS[:2])
    assert "Domanda: CUP?" in prompt
    assert "[1] scadenze" in prompt and "[2] spese ammissibili" in prompt
