from types import SimpleNamespace

import pytest
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from ragcourse.judge import LangChainJudge, deepeval_table, judge_model


class Verdict(BaseModel):
    score: float


class FakeChat(GenericFakeChatModel):
    """Replies with fixed text; structured output always gives score 1."""

    def with_structured_output(self, schema, **kwargs):
        return RunnableLambda(lambda _: schema(score=1.0))


def fake_judge() -> LangChainJudge:
    return LangChainJudge(FakeChat(messages=iter([AIMessage("Sì.")])))


def test_judge_returns_plain_text_without_a_schema():
    assert fake_judge().generate("La risposta è fedele?") == "Sì."


def test_judge_fills_the_schema_deepeval_asks_for():
    assert fake_judge().generate("Voto?", schema=Verdict) == Verdict(score=1.0)


def test_judge_plugs_into_deepeval_metrics(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")  # built offline, never called
    metric = FaithfulnessMetric(model=judge_model("google_genai:gemini-3.8-flash"))
    assert metric.evaluation_model == "google_genai:gemini-3.8-flash"


def test_deepeval_table_has_one_row_per_question_and_one_column_per_metric():
    metric = SimpleNamespace(
        name="Faithfulness", score=0.5, reason="Una frase inventata."
    )
    result = SimpleNamespace(
        test_results=[SimpleNamespace(name="faq-002", metrics_data=[metric])]
    )

    assert deepeval_table(result).to_dict() == {"Faithfulness": {"faq-002": 0.5}}
    assert (
        deepeval_table(result, "reason")
        .loc["faq-002", "Faithfulness"]
        .startswith("Una")
    )


CONTEXT = ["Le microimprese possono presentare domanda. Il contributo è del 40%."]


@pytest.mark.llm  # a few small requests to the judge model
def test_faithfulness_catches_an_invented_claim():
    def faithfulness(answer: str) -> float:
        case = LLMTestCase(
            input="Le microimprese sono ammesse?",
            actual_output=answer,
            retrieval_context=CONTEXT,
        )
        metric = FaithfulnessMetric(model=judge_model(), async_mode=False)
        return metric.measure(case)

    assert faithfulness("Sì, le microimprese possono presentare domanda.") == 1.0
    assert faithfulness("Sì, e ricevono un contributo del 90%.") < 1.0
