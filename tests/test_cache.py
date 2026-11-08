import copy
import json
import sqlite3

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ragcourse import cache
from ragcourse.cache import (
    CachedEmbeddings,
    EmbeddingCache,
    SearchCache,
    cache_llm_calls,
    measured,
)


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


def test_other_attributes_come_from_the_model(tmp_path):
    embedder = cached(CountingEmbedding(size=4), tmp_path / "e.sqlite")

    assert embedder.size == 4  # e.g. the tokenizer of a local model, read on a slide
    assert copy.copy(embedder).embed_documents(["a"]) == embedder.embed_documents(["a"])


def test_a_measurement_is_taken_once_then_read_back(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "MEASURES", tmp_path / "measures.json")
    taken = []

    def measure():
        taken.append("now")
        return 12.5

    assert measured("ms per question", measure) == 12.5
    assert measured("ms per question", measure) == 12.5  # e.g. the next render
    assert taken == ["now"]


def search_key(index, query="spese", k=5, mode="dense", filter="None"):
    return json.dumps([index, mode, query, k, filter])


def test_search_cache_survives_moving_the_project(tmp_path):
    path = tmp_path / "searches.sqlite"
    old = search_key("/desktop/course/.cache/qdrant/bge-m3-1000")
    new = search_key("/laptop/course/.cache/qdrant/bge-m3-1000")
    # The database predates portable keys, as in a zip copied from the desktop.
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE searches (key TEXT PRIMARY KEY, results TEXT)")
        db.execute(
            "INSERT INTO searches VALUES (?, ?)",
            (old, json.dumps([["Spese ammissibili", {"call": "a"}, "doc-1", 0.9]])),
        )

    for _ in range(2):  # migration also survives subsequent chapter starts
        searches = SearchCache(path)
        doc, score = searches.get(new)[0]
        assert doc == Document("Spese ammissibili", metadata={"call": "a"}, id="doc-1")
        assert score == 0.9
        keys = searches.db.execute("SELECT key FROM searches").fetchall()
        assert len(keys) == 1
        assert json.loads(keys[0][0])[0] == "qdrant:bge-m3-1000"
        searches.db.close()


def test_portable_searches_keep_models_and_search_settings_separate(tmp_path):
    searches = SearchCache(tmp_path / "searches.sqlite")
    index = "/desktop/course/.cache/qdrant/e5-1000"
    result = [(Document("Spese ammissibili"), 0.9)]
    searches.put(search_key(index), result)

    assert searches.get(search_key("C:\\course\\.cache\\qdrant\\e5-1000")) == result
    for key in [
        search_key("/laptop/course/.cache/qdrant/bge-1000"),
        search_key(index, query="fattura"),
        search_key(index, k=10),
        search_key(index, mode="sparse"),
        search_key(index, filter="bando-a"),
        search_key("memory:e5-1000"),
        search_key("/custom/e5-1000"),
    ]:
        assert searches.get(key) is None
    searches.db.close()


def test_migration_keeps_existing_portable_results(tmp_path):
    path = tmp_path / "searches.sqlite"
    searches = SearchCache(path)
    current = [(Document("Updated result"), 0.8)]
    searches.put(search_key("/laptop/course/.cache/qdrant/e5-1000"), current)
    with searches.db:
        searches.db.execute(
            "INSERT INTO searches VALUES (?, ?)",
            (search_key("/desktop/course/.cache/qdrant/e5-1000"), "[]"),
        )
    searches.db.close()

    searches = SearchCache(path)
    assert searches.get(search_key("/third/course/.cache/qdrant/e5-1000")) == current
    assert searches.db.execute("SELECT count(*) FROM searches").fetchone()[0] == 1
    searches.db.close()
