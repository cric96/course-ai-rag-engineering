import numpy as np
import pytest
from langchain_core.language_models import BaseChatModel

from ragcourse.models import chat_model, embedding_model


def test_chat_model_is_chosen_by_provider_and_name(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")  # built offline, never called
    model = chat_model("google_genai:gemini-3.8-flash")

    assert isinstance(model, BaseChatModel)
    assert model.model.endswith("gemini-3.8-flash")


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
