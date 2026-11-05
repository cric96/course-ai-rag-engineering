"""Disk caches for chat model calls and index searches: a repeat is free."""

import json
import sqlite3
import threading
from pathlib import Path

from langchain_core.caches import BaseCache
from langchain_core.documents import Document
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
