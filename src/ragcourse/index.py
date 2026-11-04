"""Vector indexes saved on disk: embedded once per configuration, then reloaded."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore

from ragcourse.paths import ROOT

INDEXES = ROOT / ".cache/qdrant"


def vector_store(
    chunks: list[Document], embeddings: Embeddings, name: str, root: Path = INDEXES
) -> QdrantVectorStore:
    """A Qdrant index on disk; `name` must identify chunking and embedder."""
    path = root / name
    if not path.exists():
        # Build aside, then move into place: a crash never leaves half an index.
        building = root / f"{name}.building"
        store = QdrantVectorStore.from_documents(
            chunks, embeddings, path=str(building), collection_name=name
        )
        store.client.close()
        building.rename(path)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings, path=str(path), collection_name=name
    )
