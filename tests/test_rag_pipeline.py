import importlib.util
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

spec = importlib.util.spec_from_file_location(
    "rag_pipeline", Path(__file__).parents[1] / "scripts/rag_pipeline.py"
)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)

CHUNKS = [
    Document("Spese ammissibili: impianti.", metadata={"title": "Bando", "page": 3}),
    Document("Scadenza: 30 giugno.", metadata={"title": "Bando", "page": 7}),
]


class FixedRetriever(BaseRetriever):
    asked: list = []

    def _get_relevant_documents(self, query, *, run_manager):
        self.asked.append(query)
        return CHUNKS


def test_the_chain_answers_from_the_chunks_it_found_once():
    llm = GenericFakeChatModel(messages=iter([AIMessage("Gli impianti [1].")]))
    retriever = FixedRetriever(asked=[])

    result = pipeline.rag_chain(retriever, llm).invoke("Quali spese?")

    assert result["answer"] == "Gli impianti [1]."
    assert result["chunks"] == CHUNKS
    assert retriever.asked == ["Quali spese?"]  # one search, shared with the answer
