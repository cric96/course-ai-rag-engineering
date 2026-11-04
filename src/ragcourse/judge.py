"""DeepEval glue: any LangChain chat model as judge, results as a table."""

import pandas as pd
from deepeval.evaluate.types import EvaluationResult
from deepeval.models import DeepEvalBaseLLM
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from ragcourse.models import JUDGE_MODEL, chat_model


class LangChainJudge(DeepEvalBaseLLM):
    """A LangChain chat model speaking DeepEval's judge interface."""

    def __init__(self, chat: BaseChatModel, name: str = "langchain"):
        self.chat = chat
        super().__init__(name)

    def load_model(self) -> BaseChatModel:
        return self.chat

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            return self.chat.invoke(prompt).text
        # DeepEval says which JSON it expects: structured output fills it in.
        return self.chat.with_structured_output(schema).invoke(prompt)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is None:
            return (await self.chat.ainvoke(prompt)).text
        return await self.chat.with_structured_output(schema).ainvoke(prompt)

    def get_model_name(self) -> str:
        return self.name


def judge_model(name: str = JUDGE_MODEL) -> LangChainJudge:
    """A judge by "provider:model" name, like chat_model."""
    return LangChainJudge(chat_model(name), name)


def deepeval_table(result: EvaluationResult, field: str = "score") -> pd.DataFrame:
    """One row per test case, one column per metric: its score (or reason)."""
    return pd.DataFrame(
        {
            test.name: {
                metric.name: getattr(metric, field) for metric in test.metrics_data
            }
            for test in result.test_results
        }
    ).T
