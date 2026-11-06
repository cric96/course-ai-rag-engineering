"""Disk caches for chat calls, index searches, embeddings and reranker scores."""

import json
import sqlite3
import threading
from array import array
from collections.abc import Callable
from pathlib import Path

from langchain_core.caches import BaseCache
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_core.load import dumps, loads
from langchain_core.outputs import ChatGeneration

from ragcourse.paths import ROOT


class SQLiteCache(BaseCache):
    """Chat answers in a SQLite file, keyed by prompt and model settings."""

    def __init__(self, path: Path):
        # Shared by the threads of batch() and DeepEval: one lock guards it.
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS calls"
                " (prompt TEXT, model TEXT, answer TEXT, PRIMARY KEY (prompt, model))"
            )

    def lookup(self, prompt: str, llm_string: str) -> list[ChatGeneration] | None:
        with self.lock:
            row = self.db.execute(
                "SELECT answer FROM calls WHERE prompt = ? AND model = ?",
                (prompt, llm_string),
            ).fetchone()
        if row is None:
            return None
        # Only chat messages are stored, so only messages may be rebuilt.
        return [ChatGeneration(message=loads(row[0], allowed_objects="messages"))]

    def update(self, prompt: str, llm_string: str, return_val: list) -> None:
        answer = dumps(return_val[0].message)
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO calls VALUES (?, ?, ?)",
                (prompt, llm_string, answer),
            )

    def clear(self, **kwargs) -> None:
        with self.lock, self.db:
            self.db.execute("DELETE FROM calls")


def cache_llm_calls(path: Path = ROOT / ".cache/llm-calls.sqlite") -> None:
    """Route every chat model call through the disk cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(path))


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

    def get(self, key: str) -> list[tuple[Document, float]] | None:
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


class ScoreCache:
    """Reranker scores, and the seconds they took, in a SQLite file."""

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        with self.lock, self.db:
            self.db.execute(
                "CREATE TABLE IF NOT EXISTS scores (key TEXT PRIMARY KEY, value TEXT)"
            )

    def get(self, key: str) -> tuple[list[float], float] | None:
        with self.lock:
            row = self.db.execute(
                "SELECT value FROM scores WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        scores, seconds = json.loads(row[0])
        return scores, seconds

    def put(self, key: str, scores: list[float], seconds: float) -> None:
        with self.lock, self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO scores VALUES (?, ?)",
                (key, json.dumps([scores, seconds])),
            )


SCORES: ScoreCache | None = None  # set by cache_scores(), read by the rerankers


def cache_scores(path: Path | None = ROOT / ".cache/rerank-scores.sqlite") -> None:
    """Keep every reranker score on disk, with its time: a re-render reuses them."""
    global SCORES
    if path is None:
        SCORES = None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    SCORES = ScoreCache(path)


MEASURES = ROOT / ".cache/measures.json"  # read at each call: tests point it elsewhere


def measured(name: str, measure: Callable[[], float]) -> float:
    """A measurement (e.g. a time) taken once, then read back from disk.

    Times must not be taken again on cached results: they would come out as 0.
    """
    known = json.loads(MEASURES.read_text()) if MEASURES.exists() else {}
    if name not in known:
        known[name] = measure()
        MEASURES.parent.mkdir(parents=True, exist_ok=True)
        MEASURES.write_text(json.dumps(known, indent=1, ensure_ascii=False))
    return known[name]
