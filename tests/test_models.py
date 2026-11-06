import numpy as np
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from ragcourse import cache, models
from ragcourse.cache import CachedEmbeddings, cache_embeddings
from ragcourse.models import (
    EMBEDDERS,
    chat_model,
    embedding_model,
    query_ms,
    sparse_model,
)


def test_chat_model_is_chosen_by_provider_and_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")  # built offline, never called
    model = chat_model("google_genai:gemini-3.8-flash")

    assert isinstance(model, BaseChatModel)
    assert model.model.endswith("gemini-3.8-flash")


def test_chat_model_keeps_provider_defaults_unless_asked(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    # Google advises against changing the temperature of Gemini 3 and later.
    assert chat_model("google_genai:gemini-3.8-flash").temperature is None
    assert (
        chat_model("google_genai:gemini-3.8-flash", temperature=0.5).temperature == 0.5
    )


def test_api_embedders_are_chosen_by_provider_and_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")  # built offline, never called
    embedder = embedding_model("google_genai:gemini-embedding-2")

    assert isinstance(embedder, GoogleGenerativeAIEmbeddings)
    assert embedder.model.endswith("gemini-embedding-2")


def test_embedders_keep_their_vectors_on_disk_once_the_cache_is_on(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr(cache, "EMBEDDINGS", None)  # restored after the test
    cache_embeddings(tmp_path / "embeddings.sqlite")

    embedder = embedding_model("google_genai:gemini-embedding-2")
    assert isinstance(embedder, CachedEmbeddings)
    assert isinstance(embedder.loaded(), GoogleGenerativeAIEmbeddings)
    with pytest.raises(ValueError, match="unknown embedding model"):
        embedding_model("not-a-model")  # still at once, not at the first new text


def test_local_catalog_names_the_hugging_face_models():
    assert EMBEDDERS["e5-small"].model == "intfloat/multilingual-e5-small"
    assert {"e5-large", "bge-m3", "qwen3-0.6b"} <= set(EMBEDDERS)


def test_unknown_embedding_model_is_rejected():
    with pytest.raises(ValueError, match="unknown embedding model"):
        embedding_model("not-a-model")


@pytest.mark.slow  # downloads and runs a real model
def test_e5_puts_related_texts_closer_than_unrelated_ones():
    embedder = embedding_model("e5-small")
    query = embedder.embed_query("Posso comprare un computer con il contributo?")
    hardware, deadline = embedder.embed_documents(
        [
            "Le spese per l'acquisto di hardware sono ammissibili.",
            "La domanda va presentata entro il 2 novembre 2026.",
        ]
    )

    assert len(query) == 384
    assert np.dot(query, hardware) > np.dot(query, deadline)


def test_unknown_sparse_model_is_rejected():
    with pytest.raises(ValueError, match="unknown sparse model"):
        sparse_model("not-a-model")


@pytest.mark.slow  # downloads the BM25 vocabulary files
def test_italian_bm25_drops_stopwords_and_stems_words():
    bm25 = sparse_model("bm25")
    assert len(bm25.embed_query("Le spese ammissibili").indices) == 2  # "le" is dropped
    assert bm25.embed_query("fattura").indices == bm25.embed_query("fatture").indices


class CountingEmbedding(DeterministicFakeEmbedding):
    """Records the questions it really embeds."""

    queries: list[str] = []

    def embed_query(self, text):
        self.queries.append(text)
        return super().embed_query(text)


def test_query_time_is_measured_on_the_real_model_once(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "MEASURES", tmp_path / "measures.json")
    model = CountingEmbedding(size=4)
    monkeypatch.setattr(models, "load_embedder", lambda name: model)

    first = query_ms("e5-small", ["Chi?", "Quando?"])
    assert first >= 0
    assert model.queries == ["riscaldamento", "Chi?", "Quando?"]  # not from a cache
    assert query_ms("e5-small", ["Chi?", "Quando?"]) == first  # kept on disk
    assert len(model.queries) == 3
