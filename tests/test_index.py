import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
    SparseEmbeddings,
    SparseVector,
)

from ragcourse import cache
from ragcourse.cache import cache_searches
from ragcourse.filters import metadata_filter
from ragcourse.index import extended_store, memory_store, search_mode, vector_store

CHUNKS = [Document("Le spese sono ammissibili.", metadata={"call": "bando-a"})]


class CountingEmbedding(DeterministicFakeEmbedding):
    """Counts embeddings of the real chunk (Qdrant also embeds a dummy probe)."""

    embedded: int = 0

    def embed_documents(self, texts):
        self.embedded += texts.count(CHUNKS[0].page_content)
        return super().embed_documents(texts)


def test_index_is_built_once_then_reloaded_from_disk(tmp_path):
    embeddings = CountingEmbedding(size=8)

    first = vector_store(CHUNKS, embeddings, "t", root=tmp_path)
    first.client.close()  # e.g. the end of a slide render
    second = vector_store(CHUNKS, embeddings, "t", root=tmp_path)

    assert embeddings.embedded == 1  # the second store came from disk
    assert second.similarity_search("spese", k=1)[0].metadata["call"] == "bando-a"


class WordSparse(SparseEmbeddings):
    """Keyword vectors without a model: one dimension per (lowercase) word."""

    def embed_query(self, text):
        words = sorted({w.strip(".?").lower() for w in text.split()})
        return SparseVector(
            indices=[hash(w) % 10_000 for w in words], values=[1.0] * len(words)
        )

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


DOCS = [
    Document("Il CUP va indicato in ogni fattura.", metadata={"call": "a"}),
    Document("Le spese di personale sono ammissibili.", metadata={"call": "a"}),
]


def test_one_hybrid_index_searches_by_meaning_by_words_or_both(tmp_path):
    store = vector_store(
        DOCS,
        DeterministicFakeEmbedding(size=8),
        "h",
        root=tmp_path,
        sparse=WordSparse(),
    )
    assert store.retrieval_mode == RetrievalMode.HYBRID

    words = search_mode(store, RetrievalMode.SPARSE)
    assert (
        words.similarity_search("cup fattura", k=1)[0].page_content
        == DOCS[0].page_content
    )
    assert search_mode(store, RetrievalMode.DENSE).client is store.client  # same index


OCR = [Document("Quadri: 47 euro all'ora.", metadata={"call": "a"})]


def test_extended_index_is_a_copy_with_extra_chunks(tmp_path):
    embeddings = DeterministicFakeEmbedding(size=8)
    vector_store(DOCS, embeddings, "base", root=tmp_path).client.close()

    extended = extended_store("base", OCR, embeddings, "base-ocr", root=tmp_path)
    assert extended.client.count(extended.collection_name).count == 3
    extended.client.close()

    again = extended_store("base", OCR, embeddings, "base-ocr", root=tmp_path)
    assert (
        again.client.count(again.collection_name).count == 3
    )  # reloaded, not re-added
    again.client.close()
    original = vector_store(DOCS, embeddings, "base", root=tmp_path)
    assert original.client.count(original.collection_name).count == 2  # untouched


@pytest.fixture
def asked(monkeypatch):
    """The searches that reach Qdrant; the search cache starts off."""
    monkeypatch.setattr(cache, "SEARCHES", None)  # restored after each test
    queries = []
    search = QdrantVectorStore.similarity_search_with_score

    def counting(self, query, *args, **kwargs):
        queries.append(query)
        return search(self, query, *args, **kwargs)

    monkeypatch.setattr(QdrantVectorStore, "similarity_search_with_score", counting)
    return queries


def test_a_search_reaches_the_index_once_then_comes_from_disk(tmp_path, asked):
    cache_searches(tmp_path / "searches.sqlite")
    store = vector_store(DOCS, DeterministicFakeEmbedding(size=8), "c", root=tmp_path)
    retriever = store.as_retriever(search_kwargs={"k": 2})
    first = retriever.invoke("fattura", filter=metadata_filter(call="a"))

    assert retriever.invoke("fattura", filter=metadata_filter(call="a")) == first
    cache_searches(tmp_path / "searches.sqlite")  # e.g. the next render
    assert retriever.invoke("fattura", filter=metadata_filter(call="a")) == first
    assert asked == ["fattura"]


def test_the_cache_tells_apart_mode_k_filter_and_index(tmp_path, asked):
    cache_searches(tmp_path / "searches.sqlite")
    fake = DeterministicFakeEmbedding(size=8)
    store = vector_store(DOCS, fake, "h", root=tmp_path, sparse=WordSparse())
    searches = [
        lambda: store.similarity_search("fattura", k=1),
        lambda: store.similarity_search("fattura", k=2),
        lambda: search_mode(store, RetrievalMode.SPARSE).similarity_search(
            "fattura", k=1
        ),
        lambda: store.similarity_search(
            "fattura", k=1, filter=metadata_filter(call="b")
        ),
    ]
    for search in searches * 2:  # the second round comes from disk
        search()
    store.client.close()
    # A copy of the index keeps the collection's name, not the results.
    extended = extended_store(
        "h", OCR, fake, "h-ocr", root=tmp_path, sparse=WordSparse()
    )
    assert len(extended.similarity_search("fattura", k=3)) == 3

    assert len(asked) == len(searches) + 1


def test_searches_are_not_cached_unless_asked(tmp_path, asked):
    store = vector_store(DOCS, DeterministicFakeEmbedding(size=8), "n", root=tmp_path)
    store.similarity_search("fattura", k=1)
    store.similarity_search("fattura", k=1)

    assert asked == ["fattura", "fattura"]


def test_an_index_in_memory_keeps_its_searches_on_disk_by_name(tmp_path, asked):
    cache_searches(tmp_path / "searches.sqlite")
    fake = DeterministicFakeEmbedding(size=8)
    first = memory_store(DOCS, fake, "e5-1000").similarity_search("fattura", k=1)

    again = memory_store(DOCS, fake, "e5-1000")  # rebuilt in the next chapter
    assert again.similarity_search("fattura", k=1) == first
    memory_store(DOCS, fake, "other").similarity_search("fattura", k=1)
    assert asked == ["fattura", "fattura"]  # only the other index was searched again
