import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from ragcourse.filters import metadata_filter

spec = importlib.util.spec_from_file_location(
    "context_recall_k",
    Path(__file__).parents[1] / "scripts/compare_context_recall_k.py",
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


@pytest.fixture
def saved_searches(tmp_path):
    questions = [
        {
            "id": "faq-1",
            "question": "Who can apply?",
            "answer": "Researchers.",
            "call": "a",
            "gold": [{"page": 1}],
        }
    ]
    path = tmp_path / "searches.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE searches (key TEXT PRIMARY KEY, results TEXT)")
        for k in [5, 10, 20]:
            key = json.dumps(
                [
                    "memory:e5-small-1000",
                    "dense",
                    questions[0]["question"],
                    k,
                    repr(metadata_filter(call="a")),
                ]
            )
            chunks = [[f"Chunk {i}", {"page": i}, f"id-{i}", 0.8] for i in range(k)]
            db.execute("INSERT INTO searches VALUES (?, ?)", (key, json.dumps(chunks)))
    return questions, path


def test_each_k_uses_its_own_saved_context_and_the_same_expected_answer(saved_searches):
    variants = experiment.saved_contexts(*saved_searches)

    for k, records in variants.items():
        assert records[0]["expected"] == "Researchers."
        assert len(records[0]["chunks"]) == k
        assert records[0]["chunks"][-1]["text"] == f"Chunk {k - 1}"
        texts = [f"Chunk {i}" for i in range(k)]
        assert (
            records[0]["context_sha256"]
            == hashlib.sha256(
                json.dumps(texts, ensure_ascii=False).encode()
            ).hexdigest()
        )


def test_missing_context_stops_instead_of_retrieving_or_generating_again(
    saved_searches,
):
    questions, path = saved_searches
    with sqlite3.connect(path) as db:
        db.execute("DELETE FROM searches WHERE key LIKE '%20,%'")

    with pytest.raises(ValueError, match="k=20"):
        experiment.saved_contexts(questions, path)
