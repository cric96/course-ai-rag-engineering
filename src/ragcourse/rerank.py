"""Reranking: fetch many chunks, then let a finer, slower model reorder them."""

from collections.abc import Callable

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel

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
    from sentence_transformers import CrossEncoder  # heavy: models group only

    model = CrossEncoder(RERANKERS[name])

    def score(query: str, docs: list[Document]) -> list[float]:
        return model.predict([(query, doc.page_content) for doc in docs]).tolist()

    return score


def decision_model(name: str = "LiquidAI/d1-2B") -> Scorer:
    """A decision model: P(yes) of "does this passage help answer the question?"."""
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


def llm_listwise(chat: BaseChatModel) -> Scorer:
    """An LLM reads all candidates and orders them; the order becomes a score."""

    def score(query: str, docs: list[Document]) -> list[float]:
        numbered = "\n\n".join(
            f"[{n}] {doc.page_content}" for n, doc in enumerate(docs, 1)
        )
        prompt = (
            f"Domanda: {query}\n\nPassaggi:\n\n{numbered}\n\n"
            "Ordina i numeri dei passaggi dal più utile al meno utile per rispondere."
        )
        order = chat.with_structured_output(Ranking).invoke(prompt).order
        rank = {n: i for i, n in enumerate(order)}
        # Forgotten candidates share the last place.
        return [
            float(len(docs) - rank.get(n, len(docs))) for n in range(1, len(docs) + 1)
        ]

    return score
