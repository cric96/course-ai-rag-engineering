"""Dense retrieval: compare the question vector with every chunk vector."""

from dataclasses import dataclass

import numpy as np

from ragcourse.chunking import Chunk
from ragcourse.embedding import Embedder


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk and its similarity to the question."""

    chunk: Chunk
    score: float


class DenseIndex:
    """All chunk vectors in one matrix: exact brute-force search, fast enough here."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.vectors = embedder.embed_documents([chunk.text for chunk in chunks])

    def search(self, question: str, k: int = 5) -> list[Hit]:
        """The k chunks most similar to the question, best first."""
        scores = self.vectors @ self.embedder.embed_query(question)  # cosine similarity
        best = np.argsort(-scores)[:k]
        return [Hit(self.chunks[i], float(scores[i])) for i in best]
