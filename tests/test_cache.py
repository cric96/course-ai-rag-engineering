import pytest
from langchain_core.globals import set_llm_cache
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from ragcourse.cache import cache_llm_calls


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
