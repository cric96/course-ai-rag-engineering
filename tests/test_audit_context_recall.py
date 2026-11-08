import copy
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest
from langchain_core.documents import Document

from ragcourse.filters import metadata_filter
from ragcourse.generation import format_context

spec = importlib.util.spec_from_file_location(
    "audit_context_recall",
    Path(__file__).parents[1] / "scripts/audit_context_recall.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def saved_case(tmp_path):
    metadata = {"url": "https://example.org/bando.pdf", "page": 2, "title": "Bando"}
    text = "Exactly the original chunk."
    question = {
        "id": "faq-a",
        "question": "Who?",
        "answer": "A researcher.",
        "call": "a",
        "gold": [{"url": metadata["url"], "page": 2}],
    }
    report = {
        "questions": [
            {
                "id": "faq-a",
                "question": "Who?",
                "expected": "A researcher.",
                "context": format_context([Document(text, metadata=metadata)]),
            }
        ],
        "runs": {
            "baseline": {
                "records": [
                    {
                        "id": "faq-a",
                        "answer": "Generated answer",
                        "scores": {"richiamo del contesto": 0},
                        "reasons": {"richiamo del contesto": "Missing information"},
                    }
                ]
            }
        },
    }
    path = tmp_path / "searches.sqlite"
    key = json.dumps(
        ["qdrant:e5-small-1000", "dense", "Who?", 5, repr(metadata_filter(call="a"))]
    )
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE searches (key TEXT PRIMARY KEY, results TEXT)")
        db.execute(
            "INSERT INTO searches VALUES (?, ?)",
            (key, json.dumps([[text, metadata, "doc-1", 0.8]])),
        )
    return report, path, [question]


def test_audit_restores_the_exact_chunk_and_keeps_the_old_score(saved_case):
    cases = audit.load_cases(*saved_case)

    assert len(cases) == 1
    assert cases[0]["chunks"][0]["text"] == "Exactly the original chunk."
    assert cases[0]["gold_chunk_indices"] == [0]
    assert cases[0]["original_score"] == 0
    assert cases[0]["answer"] == "Generated answer"


@pytest.mark.parametrize("change", ["context", "expected"])
def test_modified_inputs_are_rejected_instead_of_silently_rejudged(saved_case, change):
    report, path, questions = saved_case
    report = copy.deepcopy(report)
    report["questions"][0][change] = "Changed input"

    with pytest.raises(ValueError, match="differs"):
        audit.load_cases(report, path, questions)


def test_gold_page_is_a_selection_filter_not_a_replacement_context(saved_case):
    report, path, questions = saved_case
    questions[0]["gold"][0]["page"] = 99

    assert audit.load_cases(report, path, questions) == []
