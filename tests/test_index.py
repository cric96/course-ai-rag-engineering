from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import RetrievalMode, SparseEmbeddings, SparseVector

from ragcourse.index import extended_store, search_mode, vector_store

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
