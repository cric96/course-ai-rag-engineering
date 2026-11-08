import importlib.util
from pathlib import Path

from langchain_core.documents import Document

spec = importlib.util.spec_from_file_location(
    "evaluate_improvements",
    Path(__file__).parents[1] / "scripts/evaluate_improvements.py",
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_faq_taxonomy_partitions_the_50_sample_questions():
    ids = [f"faq-{i:03d}" for i in range(1, 51)]
    grouped = experiment.classify_questions(ids)
    assert set(grouped) == {"explicit", "deductive", "external"}
    assert sum(len(v) for v in grouped.values()) == len(ids)


def test_format_context_with_headers_keeps_citations_and_titles():
    docs = [
        Document(
            "Art. 3.2 contratti ammessi", metadata={"title": "Avviso.pdf", "page": 10}
        )
    ]
    text = experiment.context_text(docs)
    assert "[1] Avviso.pdf, p. 10" in text
    assert "Art. 3.2 contratti ammessi" in text
