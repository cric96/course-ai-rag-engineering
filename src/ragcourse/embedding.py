"""Embedders turn texts into vectors: similar meaning, nearby vectors."""

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """What retrieval needs from any embedding model, local or remote."""

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class SentenceEmbedder:
    """An open model run locally with sentence-transformers (GPU when available)."""

    def __init__(self, name: str, query_prefix: str = "", document_prefix: str = ""):
        # Imported here: PyTorch is heavy and CI does not install it.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name)
        # Some models are trained with markers, e.g. "query: " and "passage: ".
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.document_prefix + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        # Unit-length vectors: their dot product is the cosine similarity.
        return self.model.encode(texts, normalize_embeddings=True, batch_size=64)
