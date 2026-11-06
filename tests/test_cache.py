import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ragcourse.cache import CachedEmbeddings, EmbeddingCache, cache_llm_calls


@pytest.fixture
def cache_file(tmp_path):
    yield tmp_path / "llm.sqlite"
    set_llm_cache(None)  # other tests must call their models for real


def fake_model() -> GenericFakeChatModel:
    return GenericFakeChatModel(
        messages=iter([AIMessage("prima"), AIMessage("seconda")])
    )


def test_an_identical_call_is_served_from_the_cache(cache_file):
    cache_llm_calls(cache_file)
    model = fake_model()

    assert model.invoke("Chi?").text == "prima"
    assert model.invoke("Chi?").text == "prima"
    assert model.invoke("Quando?").text == "seconda"  # a new prompt reaches the model


def test_the_cache_survives_a_restart(cache_file):
    cache_llm_calls(cache_file)
    model = fake_model()
    model.invoke("Chi?")

    cache_llm_calls(cache_file)  # e.g. the next render of the slides
    assert model.invoke("Chi?").text == "prima"


class CountingEmbedding(DeterministicFakeEmbedding):
    """Records the texts it really embeds; questions get their own vectors, as in e5."""

    embedded: list[str] = []

    def embed_documents(self, texts):
        self.embedded += texts
        return super().embed_documents(texts)

    def embed_query(self, text):
        self.embedded.append(f"query: {text}")
        return super().embed_query(f"query: {text}")


def cached(model: Embeddings, path, name: str = "fake") -> CachedEmbeddings:
    return CachedEmbeddings(lambda: model, name, EmbeddingCache(path))


def test_each_text_is_embedded_once_then_read_from_disk(tmp_path):
    model = CountingEmbedding(size=4)
    vectors = cached(model, tmp_path / "e.sqlite").embed_documents(["a", "b"])

    again = cached(model, tmp_path / "e.sqlite")  # e.g. the next render of the slides
    assert again.embed_documents(["b", "a"]) == [vectors[1], vectors[0]]  # exact floats
    assert again.embed_documents(["c"]) == model.embed_documents(["c"])
    assert model.embedded == ["a", "b", "c", "c"]  # only "c" was new


def test_questions_passages_and_models_are_cached_apart(tmp_path):
    model = CountingEmbedding(size=4)
    embedder = cached(model, tmp_path / "e.sqlite")
    passage = embedder.embed_documents(["spese"])[0]

    assert embedder.embed_query("spese") != passage  # e5 marks questions differently
    cached(model, tmp_path / "e.sqlite", name="other").embed_documents(["spese"])
    assert model.embedded == ["spese", "query: spese", "spese"]


def test_the_model_is_loaded_only_for_new_texts(tmp_path):
    cached(CountingEmbedding(size=4), tmp_path / "e.sqlite").embed_documents(["a"])
    loads = []

    def load():
        loads.append("model")
        return CountingEmbedding(size=4)

    again = CachedEmbeddings(load, "fake", EmbeddingCache(tmp_path / "e.sqlite"))
    again.embed_documents(["a"])
    assert loads == []  # every vector came from disk: no model in memory
    again.embed_documents(["b"])
    assert loads == ["model"]
