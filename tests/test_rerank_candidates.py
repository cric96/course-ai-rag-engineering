import importlib.util
import json
import sqlite3
from pathlib import Path

from langchain_core.documents import Document

spec = importlib.util.spec_from_file_location(
    "compare_rerank_candidates",
    Path(__file__).parents[1] / "scripts/compare_rerank_candidates.py",
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_page_ranking_keeps_first_chunk_of_each_page():
    docs = [
        Document(str(i), metadata={"url": "bando.pdf", "page": page})
        for i, page in enumerate([3, 3, 8, 3, 2])
    ]
    assert experiment.pages_run(docs) == {
        "bando.pdf#page=3": 3,
        "bando.pdf#page=8": 2,
        "bando.pdf#page=2": 1,
    }


def test_cost_excludes_preexisting_cached_calls():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE calls (answer TEXT)")
    for tokens_in, tokens_out in [(9000000, 9000000), (1000000, 2000000)]:
        connection.execute(
            "INSERT INTO calls VALUES (?)",
            (
                json.dumps(
                    {
                        "kwargs": {
                            "usage_metadata": {
                                "input_tokens": tokens_in,
                                "output_tokens": tokens_out,
                            }
                        }
                    }
                ),
            ),
        )
    usage = experiment.new_api_usage(connection, after=1)
    assert usage["new_cached_calls"] == 1
    assert usage["input_tokens"] == 1000000
    assert usage["output_tokens_including_thinking"] == 2000000
    assert usage["estimated_usd"] == 8.25
