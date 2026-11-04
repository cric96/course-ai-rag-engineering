from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_qdrant import QdrantVectorStore

from ragcourse.filters import metadata_filter


def test_metadata_filter_searches_only_the_chunks_of_one_call():
    docs = [
        Document("Le spese sono ammissibili.", metadata={"call": "bando-a"}),
        Document("Le spese sono ammissibili.", metadata={"call": "bando-b"}),
    ]
    store = QdrantVectorStore.from_documents(
        docs,
        DeterministicFakeEmbedding(size=8),
        location=":memory:",
        collection_name="t",
    )
    retriever = store.as_retriever(search_kwargs={"k": 2})

    found = retriever.invoke("spese", filter=metadata_filter(call="bando-b"))

    assert [doc.metadata["call"] for doc in found] == ["bando-b"]
