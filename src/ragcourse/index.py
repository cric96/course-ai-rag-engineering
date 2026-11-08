"""Vector indexes saved on disk: embedded once per configuration, then reloaded."""

import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings

from ragcourse.cache.searches import CachedQdrant, index_client
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
    return open_store(path, name, embeddings, sparse)


def open_store(
    path: Path,
    collection: str,
    embeddings: Embeddings,
    sparse: SparseEmbeddings | None = None,
) -> CachedQdrant:
    """An index folder built earlier; with cached searches, it opens only on a miss."""
    client = index_client(path)
    store = CachedQdrant(
        client=client,
        collection_name=collection,
        embedding=embeddings,
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID if sparse else RetrievalMode.DENSE,
        validate_collection_config=False,  # checked when built; it opens the folder
    )
    store.index = str(path)  # the folder keys the cached searches
    return store


def memory_store(
    chunks: list[Document], embeddings: Embeddings, name: str
) -> QdrantVectorStore:
    """A Qdrant index in memory, built at each call; `name` keys its cached searches.

    `name` must identify chunking and model, as in vector_store().
    """
    store = CachedQdrant.from_documents(
        chunks, embeddings, location=":memory:", collection_name="fesr"
    )
    store.index = f"memory:{name}"  # never a folder: no clash with the indexes on disk
    return store


def search_mode(store: QdrantVectorStore, mode: RetrievalMode) -> QdrantVectorStore:
    """The same hybrid index, searched by meaning (DENSE), words (SPARSE) or both."""
    same = CachedQdrant(
        client=store.client,
        collection_name=store.collection_name,
        embedding=store.embeddings,
        sparse_embedding=store.sparse_embeddings,
        retrieval_mode=mode,
        validate_collection_config=False,  # the same collection: checking opens it
    )
    same.index = getattr(store, "index", "")  # unknown index: not cached
    return same


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
    path = root / name
    if not path.exists():
        building = root / f"{name}.building"
        shutil.rmtree(building, ignore_errors=True)  # leftovers of a crashed run
        shutil.copytree(root / base, building)
        store = open_store(building, base, embeddings, sparse)
        store.add_documents(extra)
        store.client.close()
        building.rename(path)
    # The copy keeps the collection's name; its own folder keys its searches.
    return open_store(path, base, embeddings, sparse)
