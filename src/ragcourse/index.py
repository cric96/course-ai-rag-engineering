"""Vector indexes saved on disk: embedded once per configuration, then reloaded."""

import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings

from ragcourse.paths import ROOT

INDEXES = ROOT / ".cache/qdrant"


def vector_store(
    chunks: list[Document],
    embeddings: Embeddings,
    name: str,
    root: Path = INDEXES,
    sparse: SparseEmbeddings | None = None,
) -> QdrantVectorStore:
    """A Qdrant index on disk; `name` must identify chunking and models.

    With `sparse` (e.g. BM25) every chunk also gets a keyword vector: hybrid search.
    """
    mode = RetrievalMode.HYBRID if sparse else RetrievalMode.DENSE
    path = root / name
    if not path.exists():
        # Build aside, then move into place: a crash never leaves half an index.
        building = root / f"{name}.building"
        store = QdrantVectorStore.from_documents(
            chunks,
            embeddings,
            sparse_embedding=sparse,
            retrieval_mode=mode,
            path=str(building),
            collection_name=name,
        )
        store.client.close()
        building.rename(path)
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse,
        retrieval_mode=mode,
        path=str(path),
        collection_name=name,
    )


def search_mode(store: QdrantVectorStore, mode: RetrievalMode) -> QdrantVectorStore:
    """The same hybrid index, searched by meaning (DENSE), words (SPARSE) or both."""
    return QdrantVectorStore(
        client=store.client,
        collection_name=store.collection_name,
        embedding=store.embeddings,
        sparse_embedding=store.sparse_embeddings,
        retrieval_mode=mode,
    )


def extended_store(
    base: str,
    extra: list[Document],
    embeddings: Embeddings,
    name: str,
    root: Path = INDEXES,
    sparse: SparseEmbeddings | None = None,
) -> QdrantVectorStore:
    """A copy of index `base` with `extra` chunks added (e.g. OCR text), kept on disk.

    Copying the folder is much faster than embedding the whole corpus again.
    """
    mode = RetrievalMode.HYBRID if sparse else RetrievalMode.DENSE
    path = root / name

    def open_at(folder: Path) -> QdrantVectorStore:
        return QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            sparse_embedding=sparse,
            retrieval_mode=mode,
            path=str(folder),
            collection_name=base,  # the copy keeps the collection's name
        )

    if not path.exists():
        building = root / f"{name}.building"
        shutil.rmtree(building, ignore_errors=True)  # leftovers of a crashed run
        shutil.copytree(root / base, building)
        store = open_at(building)
        store.add_documents(extra)
        store.client.close()
        building.rename(path)
    return open_at(path)
