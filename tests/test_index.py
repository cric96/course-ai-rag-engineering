from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ragcourse.index import vector_store

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
