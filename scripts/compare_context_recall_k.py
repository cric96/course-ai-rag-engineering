"""Measure context recall for the chapter's saved 5-, 10- and 20-chunk contexts."""

import argparse
import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import ragcourse  # noqa: F401

# isort: split

import pandas as pd
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from ragcourse.cache import cache_llm_calls
from ragcourse.filters import metadata_filter
from ragcourse.io import read_jsonl
from ragcourse.judge import LangChainJudge
from ragcourse.models import JUDGE_MODEL, chat_model
from ragcourse.paths import DATA, ROOT


def saved_contexts(questions: list[dict], path: Path) -> dict[int, list[dict]]:
    """Use the exact in-memory-index search results saved by chapter six."""
    variants = {}
    with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as db:
        for k in [5, 10, 20]:
            records = []
            for question in questions:
                key = json.dumps(
                    [
                        "memory:e5-small-1000",
                        "dense",
                        question["question"],
                        k,
                        repr(metadata_filter(call=question["call"])),
                    ]
                )
                row = db.execute(
                    "SELECT results FROM searches WHERE key=?", (key,)
                ).fetchone()
                if row is None:
                    raise ValueError(f"Missing chapter context: {question['id']} k={k}")
                chunks = [
                    {
                        "text": text,
                        "metadata": metadata,
                        "id": doc_id,
                        "similarity": score,
                    }
                    for text, metadata, doc_id, score in json.loads(row[0])
                ]
                records.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "expected": question["answer"],
                        "chunks": chunks,
                        "context_sha256": hashlib.sha256(
                            json.dumps(
                                [c["text"] for c in chunks], ensure_ascii=False
                            ).encode()
                        ).hexdigest(),
                    }
                )
            variants[k] = records
    return variants


async def run(args):
    questions = (
        pd.DataFrame(
            [q for q in read_jsonl(DATA / "eval/faq_questions.jsonl") if q["gold"]]
        )
        .sample(50, random_state=0)
        .to_dict("records")
    )
    variants = saved_contexts(questions, args.search_cache)
    report = {
        "questions": len(questions),
        "sample_seed": 0,
        "judge": JUDGE_MODEL,
        "index": "memory:e5-small-1000",
        "generated_answers": False,
        "metric": "ContextualRecallMetric; standard expected output attribution",
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(args.output)

    cache_llm_calls()
    judge = LangChainJudge(
        chat_model(JUDGE_MODEL, timeout=120, max_retries=2), JUDGE_MODEL
    )
    gate = asyncio.Semaphore(args.concurrency)
    save()

    async def measure(k, record):
        async with gate:
            metric = ContextualRecallMetric(model=judge)
            case = LLMTestCase(
                input=record["question"],
                expected_output=record["expected"],
                retrieval_context=[c["text"] for c in record["chunks"]],
            )
            record["score"] = await metric.a_measure(case, _show_indicator=False)
            record["reason"] = metric.reason
            record["verdicts"] = [v.model_dump(mode="json") for v in metric.verdicts]
            return k, record["id"], record["score"]

    jobs = [measure(k, record) for k, records in variants.items() for record in records]
    for done, future in enumerate(asyncio.as_completed(jobs), start=1):
        k, ident, score = await future
        save()
        print(
            f"Measured {done}/{len(jobs)}: {ident} k={k} recall={score:.3f}", flush=True
        )
    report["summary"] = {
        str(k): {
            "mean": sum(r["score"] for r in records) / len(records),
            "zero": sum(r["score"] == 0 for r in records),
            "count": len(records),
        }
        for k, records in variants.items()
    }
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-cache", type=Path, default=ROOT / ".cache/searches.sqlite"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/context-recall-k.json"
    )
    parser.add_argument("--concurrency", type=int, default=6)
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
