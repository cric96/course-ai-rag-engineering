"""Vector indexes saved on disk: embedded once per configuration, then reloaded."""

import json
import shutil
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, SparseEmbeddings

from ragcourse import cache
from ragcourse.paths import ROOT

INDEXES = ROOT / ".cache/qdrant"


class CachedQdrant(QdrantVectorStore):
    """Qdrant whose searches go through the disk cache, once cache_searches() is on."""

    index = ""  # the index folder: with mode, query, k and filter, it keys the cache

    def similarity_search_with_score(self, query, k=4, filter=None, **options):
        searches = cache.SEARCHES
        # Only plain searches (query, k, filter) of a known index are cached.
        if searches is None or not self.index or any(options.values()):
            return super().similarity_search_with_score(query, k, filter, **options)
        key = json.dumps(
            [self.index, self.retrieval_mode.value, query, k, repr(filter)]
        )
        found = searches.get(key)
        if found is None:
            found = super().similarity_search_with_score(query, k, filter)
            searches.put(key, found)
        return found


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
    store = CachedQdrant.from_existing_collection(
        embedding=embeddings,
        sparse_embedding=sparse,
        retrieval_mode=mode,
        path=str(path),
        collection_name=name,
    )
    store.index = str(path)
    return store


def search_mode(store: QdrantVectorStore, mode: RetrievalMode) -> QdrantVectorStore:
    """The same hybrid index, searched by meaning (DENSE), words (SPARSE) or both."""
    same = CachedQdrant(
        client=store.client,
        collection_name=store.collection_name,
        embedding=store.embeddings,
        sparse_embedding=store.sparse_embeddings,
        retrieval_mode=mode,
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
    mode = RetrievalMode.HYBRID if sparse else RetrievalMode.DENSE
    path = root / name

    def open_at(folder: Path) -> CachedQdrant:
        return CachedQdrant.from_existing_collection(
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
    store = open_at(path)
    store.index = str(path)  # its own folder: same collection name, other chunks
    return store
