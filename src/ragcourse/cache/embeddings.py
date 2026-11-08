"""Embedding vectors kept on disk: each text is embedded once per model."""

import json
import sqlite3
import threading
from array import array
from collections.abc import Callable
from pathlib import Path

from langchain_core.embeddings import Embeddings

from ragcourse.paths import ROOT


class EmbeddingCache:
    """Embedding vectors in a SQLite file, keyed by model, kind and text."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB)"
            )

    def get(self, key: str) -> list[float] | None:
        with self.lock:
            row = self.db.execute(
                "SELECT vector FROM vectors WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else array("d", row[0]).tolist()

    def put(self, vectors: dict[str, list[float]]) -> None:
        # Doubles: every float comes back exactly as the model gave it.
        rows = [(key, array("d", vector).tobytes()) for key, vector in vectors.items()]
        with self.lock, self.db:
            self.db.executemany("INSERT OR REPLACE INTO vectors VALUES (?, ?)", rows)


class CachedEmbeddings(Embeddings):
    """An embedder whose vectors are kept on disk: each text is embedded once."""

    def __init__(
        self, load: Callable[[], Embeddings], name: str, vectors: EmbeddingCache
    ):
        self.load, self.name, self.vectors = load, name, vectors
        self.model: Embeddings | None = None  # loaded only when a text is new

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.cached(
            texts, "passage", lambda new: self.loaded().embed_documents(new)
        )

    def embed_query(self, text: str) -> list[float]:
        return self.cached(
            [text], "query", lambda new: [self.loaded().embed_query(new[0])]
        )[0]

    def __getattr__(self, name: str):
        # Called only for what this class lacks, e.g. the tokenizer of a local model.
        if "load" not in vars(self):  # not built yet, e.g. while being copied
            raise AttributeError(name)
        return getattr(self.loaded(), name)

    def loaded(self) -> Embeddings:
        if self.model is None:
            self.model = self.load()
        return self.model

    def cached(self, texts: list[str], kind: str, embed) -> list[list[float]]:
        """Vectors from disk; the missing texts are embedded together, then stored."""
        keys = [json.dumps([self.name, kind, text]) for text in texts]
        found = {key: self.vectors.get(key) for key in keys}
        missing = [i for i, key in enumerate(keys) if found[key] is None]
        if missing:
            new = embed([texts[i] for i in missing])
            fresh = {keys[i]: vector for i, vector in zip(missing, new, strict=True)}
            self.vectors.put(fresh)
            found |= fresh
        return [found[key] for key in keys]


EMBEDDINGS: EmbeddingCache | None = None  # set by cache_embeddings(), read by models


def cache_embeddings(path: Path | None = ROOT / ".cache/embeddings.sqlite") -> None:
    """Keep every embedding vector on disk: a text is embedded once. None: off."""
    global EMBEDDINGS
    if path is None:
        EMBEDDINGS = None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    EMBEDDINGS = EmbeddingCache(path)
