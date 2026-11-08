"""Reranker scores, and the seconds they took, kept on disk."""

import json
import sqlite3
import threading
import time
from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document

from ragcourse.paths import ROOT

# The reranker interface (ragcourse.rerank): question and chunks in, scores out.
Scorer = Callable[[str, list[Document]], list[float]]


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


class CachedScorer:
    """A reranker whose scores, and the seconds each took, are kept on disk."""

    def __init__(self, load: Callable[[], Scorer], name: str, scores: ScoreCache):
        self.load, self.name, self.scores = load, name, scores
        self.scorer: Scorer | None = None  # loaded only for new candidates
        self.seconds: list[float] = []  # one per reranking, as first measured

    def __call__(self, query: str, docs: list[Document]) -> list[float]:
        key = json.dumps([self.name, query, [doc.page_content for doc in docs]])
        found = self.scores.get(key)
        if found is None:
            if self.scorer is None:
                self.scorer = self.load()
            start = time.perf_counter()
            found = (self.scorer(query, docs), time.perf_counter() - start)
            self.scores.put(key, *found)
        scores, seconds = found
        self.seconds.append(seconds)
        return scores


def cached_scorer(name: str, load: Callable[[], Scorer]) -> Scorer:
    """The scorer `load()` gives, its scores kept on disk once cache_scores() is on."""
    if SCORES is None:
        return load()
    return CachedScorer(load, name, SCORES)
