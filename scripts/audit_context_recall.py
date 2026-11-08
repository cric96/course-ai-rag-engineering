"""Rejudge zero-recall cases from their exact saved chunks, bypassing LLM caches."""

import argparse
import asyncio
import hashlib
import json
import sqlite3
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import ragcourse  # noqa: F401

# isort: split

from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase
from langchain_core.documents import Document

from ragcourse.filters import metadata_filter
from ragcourse.generation import format_context
from ragcourse.io import read_jsonl
from ragcourse.judge import LangChainJudge
from ragcourse.models import JUDGE_MODEL, chat_model
from ragcourse.paths import DATA, ROOT


def load_cases(report: dict, search_cache: Path, questions: list[dict]) -> list[dict]:
    """Recover original chunks and select zero scores with a retrieved gold page."""
    questions = {q["id"]: q for q in questions}
    inputs = {q["id"]: q for q in report["questions"]}
    selected = []
    with sqlite3.connect(f"file:{search_cache.resolve()}?mode=ro", uri=True) as db:
        for record in report["runs"]["baseline"]["records"]:
            if record["scores"]["richiamo del contesto"] != 0:
                continue
            question = questions[record["id"]]
            saved = inputs[record["id"]]
            if (saved["question"], saved["expected"]) != (
                question["question"],
                question["answer"],
            ):
                raise ValueError(f"Original FAQ differs: {question['id']}")
            key = json.dumps(
                [
                    "qdrant:e5-small-1000",
                    "dense",
                    question["question"],
                    5,
                    repr(metadata_filter(call=question["call"])),
                ]
            )
            row = db.execute(
                "SELECT results FROM searches WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Original retrieval missing: {question['id']}")
            chunks = [
                {"text": text, "metadata": metadata, "id": doc_id, "similarity": score}
                for text, metadata, doc_id, score in json.loads(row[0])
            ]
            restored = [Document(c["text"], metadata=c["metadata"]) for c in chunks]
            if format_context(restored) != inputs[question["id"]]["context"]:
                raise ValueError(f"Original context differs: {question['id']}")
            gold = {(g["url"], g["page"]) for g in question["gold"]}
            gold_chunks = [
                i
                for i, c in enumerate(chunks)
                if (c["metadata"]["url"], c["metadata"]["page"]) in gold
            ]
            if gold_chunks:
                selected.append(
                    {
                        "id": question["id"],
                        "question": question["question"],
                        "expected": question["answer"],
                        "gold": question["gold"],
                        "answer": record["answer"],
                        "original_score": 0,
                        "original_reason": record["reasons"]["richiamo del contesto"],
                        "chunks": chunks,
                        "gold_chunk_indices": gold_chunks,
                    }
                )
    return selected


def summarize(results: list[dict]) -> dict:
    grouped = defaultdict(list)
    for result in results:
        if "score" in result:
            grouped[f"{result['judge']}/{result['variant']}"].append(result["score"])
    return {
        key: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "positive": sum(value > 0 for value in values),
            "zero": sum(value == 0 for value in values),
        }
        for key, values in grouped.items()
    }


async def run(args):
    original = json.loads(args.report.read_text())
    cases = load_cases(
        original, args.search_cache, read_jsonl(DATA / "eval/faq_questions.jsonl")
    )
    if args.sample:
        cases = cases[: args.sample]
    if not cases:
        raise ValueError("No zero-recall cases with a gold-page hit")
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "llm_cache": False,
        "deepeval_result_cache": False,
        "context_mode": args.context,
        "selection": (
            "original recall zero; at least one retrieved chunk from a gold page"
        ),
        "cases": cases,
        "results": [],
        "note": "Saved chunks and original FAQ only; answers are not regenerated.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(args.output)

    models = {"flash": JUDGE_MODEL}
    judges = {
        name: LangChainJudge(
            chat_model(model, cache=False, timeout=120, max_retries=2), model
        )
        for name, model in models.items()
    }
    jobs = []
    for case in cases:
        texts = [c["text"] for c in case["chunks"]]
        if args.context == "gold-chunks":
            jobs += [
                (case, "flash", f"gold_chunk_{i + 1}", [texts[i]])
                for i in case["gold_chunk_indices"]
            ]
        else:
            jobs += [(case, judge, "original_5_chunks", texts) for judge in judges]
    save()
    print(f"Selected {len(cases)} cases; {len(jobs)} fresh evaluations", flush=True)
    gate = asyncio.Semaphore(args.concurrency)

    async def measure(case, judge, variant, context):
        async with gate:
            result = {
                "id": case["id"],
                "judge": judge,
                "model": models[judge],
                "variant": variant,
                "context_sha256": hashlib.sha256(
                    json.dumps(context).encode()
                ).hexdigest(),
            }
            metric = ContextualRecallMetric(model=judges[judge])
            test = LLMTestCase(
                input=case["question"],
                expected_output=case["expected"],
                actual_output=case["answer"],
                retrieval_context=context,
            )
            try:
                result["score"] = await metric.a_measure(test, _show_indicator=False)
                result["reason"] = metric.reason
                result["verdicts"] = [
                    v.model_dump(mode="json") for v in metric.verdicts
                ]
            except Exception as error:
                result["error"] = f"{type(error).__name__}: {error}"
            return result

    for future in asyncio.as_completed([measure(*job) for job in jobs]):
        result = await future
        report["results"].append(result)
        save()
        print(
            f"Evaluated {len(report['results'])}/{len(jobs)}: "
            f"{result['id']} {result['judge']} {result['variant']} "
            f"{result.get('score', 'ERROR')}",
            flush=True,
        )
    report["summary"] = summarize(report["results"])
    report["errors"] = sum("error" in r for r in report["results"])
    save()
    print(json.dumps(report["summary"], indent=2), flush=True)
    if report["errors"]:
        raise RuntimeError(f"Audit incomplete: {report['errors']} failed evaluations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report", type=Path, default=ROOT / ".cache/generator-comparison-argon.json"
    )
    parser.add_argument(
        "--search-cache", type=Path, default=ROOT / ".cache/searches.sqlite"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/context-recall-audit.json"
    )
    parser.add_argument("--sample", type=int)
    parser.add_argument(
        "--context", choices=["original", "gold-chunks"], default="original"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.concurrency < 1 or (args.sample is not None and args.sample < 1):
        parser.error("--concurrency and --sample must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
