"""Visual RAG: search the page pictures themselves, without reading them first."""

import uuid
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from ragcourse.index import INDEXES, open_store
from ragcourse.ocr import page_picture

# The model card advises instructions in English, also for other languages.
QUERY_PROMPT = "Find the document page that answers the question."


class PictureEmbeddings(Embeddings):
    """Qwen3-VL-Embedding: page pictures and questions in the same vector space."""

    def __init__(self, name: str = "Qwen/Qwen3-VL-Embedding-2B", batch_size: int = 4):
        from sentence_transformers import SentenceTransformer  # heavy: models group

        self.model = SentenceTransformer(name, model_kwargs={"dtype": "bfloat16"})
        self.batch_size = batch_size

    def embed_pictures(self, pages: list[dict]) -> list[list[float]]:
        pictures = [{"image": page_picture(page, dpi=100)} for page in pages]
        vectors = self.model.encode(
            pictures, batch_size=self.batch_size, normalize_embeddings=True
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(text, prompt=QUERY_PROMPT, normalize_embeddings=True)
        return vector.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]


def payload(page: dict) -> dict:
    """Same layout as LangChain's: its retrievers and filters work unchanged."""
    return {
        "page_content": f"[immagine] {page['title']}, p. {page['page']}",
        "metadata": {key: page[key] for key in ("url", "page", "title", "call")},
    }


def picture_store(
    pages: list[dict], embeddings: Embeddings, name: str, root: Path = INDEXES
) -> QdrantVectorStore:
    """A Qdrant index of page pictures, on disk; questions search it as text."""
    path = root / name
    if not path.exists():
        building = root / f"{name}.building"
        client = QdrantClient(path=str(building))
        vectors = embeddings.embed_pictures(pages)
        client.create_collection(
            name,
            vectors_config=models.VectorParams(
                size=len(vectors[0]), distance=models.Distance.COSINE
            ),
        )
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()), vector=vector, payload=payload(page)
            )
            for page, vector in zip(pages, vectors, strict=True)
        ]
        client.upsert(name, points=points)
        client.close()
        building.rename(path)
    return open_store(path, name, embeddings)  # searches cached, opened lazily
