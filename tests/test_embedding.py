import pytest

E5_SMALL = "intfloat/multilingual-e5-small"


@pytest.mark.slow  # downloads and runs a real model
def test_e5_puts_related_texts_closer_than_unrelated_ones():
    from ragcourse.embedding import SentenceEmbedder

    embedder = SentenceEmbedder(E5_SMALL, "query: ", "passage: ")
    query = embedder.embed_query("Posso comprare un computer con il contributo?")
    hardware, deadline = embedder.embed_documents(
        [
            "Le spese per l'acquisto di hardware sono ammissibili.",
            "La domanda va presentata entro il 2 novembre 2026.",
        ]
    )

    assert query.shape == (384,)
    assert query @ hardware > query @ deadline
