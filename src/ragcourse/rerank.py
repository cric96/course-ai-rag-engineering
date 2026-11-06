"""Reranking: fetch many chunks, then let a finer, slower model reorder them."""

import json
import time
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

from ragcourse import cache
from ragcourse.cache import ScoreCache

# A reranker scores how much each chunk helps answer the question: higher is better.
Scorer = Callable[[str, list[Document]], list[float]]

RERANKERS = {
    "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",  # encoder: question + chunk
    "qwen3-reranker-0.6b": "Qwen/Qwen3-Reranker-0.6B",  # decoder: logit of "yes"
}


class RerankRetriever(BaseRetriever):
    """Any retriever, then a reranker that keeps the best `top_n` chunks."""

    base: BaseRetriever
    scorer: Scorer
    top_n: int = 10

    def _get_relevant_documents(self, query, *, run_manager, **options):
        candidates = self.base.invoke(query, **options)  # options: e.g. the call filter
        scores = self.scorer(query, candidates)
        best = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)
        return [candidates[i] for i in best[: self.top_n]]


def cross_encoder(name: str) -> Scorer:
    """A Hugging Face reranker from RERANKERS, run locally."""
    return cached_scorer(name, lambda: load_cross_encoder(name))


def load_cross_encoder(name: str) -> Scorer:
    """The cross-encoder itself, loaded into memory (GPU if any)."""
    from sentence_transformers import CrossEncoder  # heavy: models group only

    model = CrossEncoder(RERANKERS[name])

    def score(query: str, docs: list[Document]) -> list[float]:
        return model.predict([(query, doc.page_content) for doc in docs]).tolist()

    return score


def decision_model(name: str = "LiquidAI/d1-2B") -> Scorer:
    """A decision model: P(yes) of "does this passage help answer the question?"."""
    return cached_scorer(name, lambda: load_decision_model(name))


def load_decision_model(name: str) -> Scorer:
    """The decision model itself, loaded into memory (GPU if any)."""
    import torch  # heavy: models group only
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        name, trust_remote_code=True, dtype=torch.bfloat16
    )
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    def score(query: str, docs: list[Document]) -> list[float]:
        question = {
            "helps": {
                "type": "noul",  # a yes/no question, answered with P(yes)
                "instructions": f"Does this passage help answer the question: {query}",
            }
        }
        outputs = model.system_one_batch([(doc.page_content, question) for doc in docs])
        return [out["answers"]["helps"]["noul"] for out in outputs]

    return score


class CachedScorer:
    """A reranker whose scores, and the seconds each took, are kept on disk."""

    def __init__(self, load: Callable[[], Scorer], name: str, scores: ScoreCache):
        self.load, self.name, self.scores = load, name, scores
        self.scorer: Scorer | None = None  # loaded only for new candidates
        self.seconds: list[float] = []  # one per reranking, as first measured

    def __call__(self, query: str, docs: list[Document]) -> list[float]:
        key = json.dumps([self.name, query, [doc.page_content for doc in docs]])
        found = self.scores.get(key)
        if found is None:
            if self.scorer is None:
                self.scorer = self.load()
            start = time.perf_counter()
            found = (self.scorer(query, docs), time.perf_counter() - start)
            self.scores.put(key, *found)
        scores, seconds = found
        self.seconds.append(seconds)
        return scores


def cached_scorer(name: str, load: Callable[[], Scorer]) -> Scorer:
    """The scorer `load()` gives, its scores kept on disk once cache_scores() is on."""
    if cache.SCORES is None:
        return load()
    return CachedScorer(load, name, cache.SCORES)


class Ranking(BaseModel):
    """The LLM's answer: candidate numbers, most useful first."""

    order: list[int]


def listwise_prompt(query: str, docs: list[Document]) -> str:
    """The question and the numbered candidates, for an LLM to put in order."""
    numbered = "\n\n".join(f"[{n}] {doc.page_content}" for n, doc in enumerate(docs, 1))
    return (
        f"Domanda: {query}\n\nPassaggi:\n\n{numbered}\n\n"
        "Ordina i numeri dei passaggi dal più utile al meno utile per rispondere."
    )


def llm_listwise(chat: BaseChatModel) -> Scorer:
    """An LLM reads all candidates and orders them; the order becomes a score."""

    def score(query: str, docs: list[Document]) -> list[float]:
        order = chat.with_structured_output(Ranking).invoke(
            listwise_prompt(query, docs)
        )
        rank = {n: i for i, n in enumerate(order.order)}
        # Forgotten candidates share the last place.
        return [
            float(len(docs) - rank.get(n, len(docs))) for n in range(1, len(docs) + 1)
        ]

    return score


class CachedRetriever(BaseRetriever):
    """Runs a slow first stage once per question; later calls reuse its chunks."""

    base: BaseRetriever
    seen: dict = {}

    def _get_relevant_documents(self, query, *, run_manager, **options):
        key = (query, repr(options))  # e.g. the call filter is part of the question
        if key not in self.seen:
            self.seen[key] = self.base.invoke(query, **options)
        return self.seen[key]
