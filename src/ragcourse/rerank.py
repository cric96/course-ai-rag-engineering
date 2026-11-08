"""Reranking: fetch many chunks, then let a finer, slower model reorder them."""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

from ragcourse.cache.scores import cached_scorer

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


def passage_text(doc: Document, with_titles: bool = False) -> str:
    """Chunk text, optionally headed by its document title and page."""
    if with_titles and "title" in doc.metadata and "page" in doc.metadata:
        return f"{doc.metadata['title']}, p. {doc.metadata['page']}\n{doc.page_content}"
    return doc.page_content


def cross_encoder(name: str, with_titles: bool = False) -> Scorer:
    """A Hugging Face reranker from RERANKERS, run locally."""
    key = f"{name}:titles" if with_titles else name
    return cached_scorer(key, lambda: load_cross_encoder(name, with_titles=with_titles))


def load_cross_encoder(name: str, with_titles: bool = False) -> Scorer:
    """The cross-encoder itself, loaded into memory (GPU if any)."""
    from sentence_transformers import CrossEncoder  # heavy: models group only

    model = CrossEncoder(RERANKERS[name])

    def score(query: str, docs: list[Document]) -> list[float]:
        pairs = [(query, passage_text(doc, with_titles=with_titles)) for doc in docs]
        return model.predict(pairs).tolist()

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


class Ranking(BaseModel):
    """The LLM's answer: candidate numbers, most useful first."""

    order: list[int]


def listwise_prompt(query: str, docs: list[Document], with_titles: bool = False) -> str:
    """The question and the numbered candidates, for an LLM to put in order."""
    numbered = "\n\n".join(
        f"[{n}] {passage_text(doc, with_titles=with_titles)}"
        for n, doc in enumerate(docs, 1)
    )
    return (
        f"Domanda: {query}\n\nPassaggi:\n\n{numbered}\n\n"
        "Ordina i numeri dei passaggi dal più utile al meno utile per rispondere."
    )


def llm_listwise(chat: BaseChatModel, with_titles: bool = False) -> Scorer:
    """An LLM reads all candidates and orders them; the order becomes a score."""

    def score(query: str, docs: list[Document]) -> list[float]:
        order = chat.with_structured_output(Ranking).invoke(
            listwise_prompt(query, docs, with_titles=with_titles)
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
