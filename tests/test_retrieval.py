import numpy as np
import pytest

from ragcourse.chunking import Chunk
from ragcourse.retrieval import DenseIndex

VOCABULARY = ["hardware", "scadenza", "domanda", "spese"]


class WordEmbedder:
    """Fake embedder: one dimension per known word, enough to test ranking."""

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vectors = np.array([[w in t.lower() for w in VOCABULARY] for t in texts], float)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def chunk(text: str, page: int) -> Chunk:
    return Chunk(text, "bando.pdf", page, "Bando")


CHUNKS = [
    chunk("La scadenza della domanda", 1),
    chunk("Spese per hardware", 2),
    chunk("Spese generali", 3),
]


def test_search_returns_the_most_similar_chunks_first():
    hits = DenseIndex(CHUNKS, WordEmbedder()).search("hardware e spese", k=2)

    assert [hit.chunk.page for hit in hits] == [2, 3]
    assert hits[0].score == pytest.approx(1.0)
