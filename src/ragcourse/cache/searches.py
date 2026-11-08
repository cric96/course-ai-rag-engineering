"""Index searches kept on disk: a repeated search never opens the index."""

import json
import sqlite3
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from ragcourse.paths import ROOT


def portable_search_key(key: str) -> str:
    """Identify course indexes independently of the project's location."""
    parts = json.loads(key)
    index = parts[0].replace("\\", "/")
    marker = "/.cache/qdrant/"
    if marker not in index:
        return key  # memory indexes and custom index folders keep their own identity
    parts[0] = "qdrant:" + index.split(marker, 1)[1]
    return json.dumps(parts)


class SearchCache:
    """Index search results in a SQLite file: the same search is read back."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS searches"
                " (key TEXT PRIMARY KEY, results TEXT)"
            )
            # Preserve existing results when the project is moved to another computer.
            keys = self.db.execute("SELECT key FROM searches").fetchall()
            for (old,) in keys:
                new = portable_search_key(old)
                if new != old:
                    self.db.execute(
                        "UPDATE OR IGNORE searches SET key = ? WHERE key = ?",
                        (new, old),
                    )
                    # If both keys exist, keep the portable entry.
                    self.db.execute("DELETE FROM searches WHERE key = ?", (old,))

    def get(self, key: str) -> list[tuple[Document, float]] | None:
        key = portable_search_key(key)
        with self.lock:
            row = self.db.execute(
                "SELECT results FROM searches WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return [
            (Document(text, metadata=metadata, id=doc_id), score)
            for text, metadata, doc_id, score in json.loads(row[0])
        ]

    def put(self, key: str, results: list[tuple[Document, float]]) -> None:
        key = portable_search_key(key)
        rows = [
            [doc.page_content, doc.metadata, doc.id, score] for doc, score in results
        ]
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO searches VALUES (?, ?)",
                (key, json.dumps(rows, ensure_ascii=False)),
            )


SEARCHES: SearchCache | None = None  # set by cache_searches(), read by the indexes


def cache_searches(path: Path | None = ROOT / ".cache/searches.sqlite") -> None:
    """Keep every index search on disk: asking the same again is free. None: off."""
    global SEARCHES
    if path is None:
        SEARCHES = None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    SEARCHES = SearchCache(path)


class LazyClient:
    """A local Qdrant folder, opened at its first use: cached searches never open it."""

    def __init__(self, path: Path):
        self.path = path
        self.opened: QdrantClient | None = None

    def __getattr__(self, name: str):
        # Called only for what this class lacks: every QdrantClient method.
        if "path" not in vars(self):  # not built yet, e.g. while being copied
            raise AttributeError(name)
        if self.opened is None:
            self.opened = QdrantClient(path=str(self.path))  # loads the whole index
        return getattr(self.opened, name)

    def close(self, **options) -> None:
        if self.opened is not None:
            self.opened.close(**options)


class CachedQdrant(QdrantVectorStore):
    """Qdrant whose searches go through the disk cache, once cache_searches() is on."""

    index = ""  # the index folder: with mode, query, k and filter, it keys the cache

    def similarity_search_with_score(self, query, k=4, filter=None, **options):
        searches = SEARCHES
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


def index_client(path: Path) -> QdrantClient:
    """A client for an index folder: opened at first use while searches are cached."""
    # Without the search cache (e.g. the chat app), open now: ready for question one.
    return LazyClient(path) if SEARCHES else QdrantClient(path=str(path))
