"""Compare Gemini and Qwen3 with 20, 30 and 50 hybrid candidates.

    uv run python scripts/compare_rerank_candidates.py

Keeps ten chunks after reranking; evaluates retrieval only, with no generation/judge.
Existing searches, scores and Gemini calls are reused. The JSON report is resumable.
"""

import argparse
import gc
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

import ragcourse  # noqa: F401

# isort: split

import numpy as np
from ranx import Run, evaluate

from ragcourse.cache import (
    cache_embeddings,
    cache_llm_calls,
    cache_scores,
    cache_searches,
)
from ragcourse.evaluation import gold_qrels, page_id
from ragcourse.filters import metadata_filter
from ragcourse.index import open_store
from ragcourse.io import read_jsonl
from ragcourse.models import CHAT_MODEL, chat_model, embedding_model, sparse_model
from ragcourse.paths import DATA, ROOT
from ragcourse.rerank import Ranking, cross_encoder, listwise_prompt

SIZES = (20, 30, 50)
METRICS = ["hit_rate@1", "hit_rate@5", "mrr@10"]
# Gemini Developer API introductory pricing, verified 2026-11-09.
INPUT_USD_PER_M = 0.75
OUTPUT_USD_PER_M = 3.75


def pages_run(docs):
    """Same page deduplication and scoring as the course's retrieval_run."""
    pages = dict.fromkeys(page_id(d.metadata["url"], d.metadata["page"]) for d in docs)
    return {p: len(pages) - rank for rank, p in enumerate(pages)}


def new_api_usage(connection, after):
    """Only newly cached Gemini calls count toward this run's estimated spend."""
    tokens_in = tokens_out = calls = 0
    for (answer,) in connection.execute(
        "SELECT answer FROM calls WHERE rowid > ?", (after,)
    ):
        usage = json.loads(answer).get("kwargs", {}).get("usage_metadata") or {}
        tokens_in += usage.get("input_tokens", 0)
        tokens_out += usage.get("output_tokens", 0)
        calls += 1
    return {
        "new_cached_calls": calls,
        "input_tokens": tokens_in,
        "output_tokens_including_thinking": tokens_out,
        "estimated_usd": (tokens_in * INPUT_USD_PER_M + tokens_out * OUTPUT_USD_PER_M)
        / 1e6,
    }


def write_artifacts(report):
    """Keep a compact, reviewable result and chart alongside the course assets."""
    import pandas as pd

    from ragcourse.viz import compare_chart

    assets = ROOT / "slides/assets"
    summary = {key: value for key, value in report.items() if key != "records"}
    summary["pricing"] = {
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
        "checked": "2026-11-09",
        "input_usd_per_million": INPUT_USD_PER_M,
        "output_usd_per_million": OUTPUT_USD_PER_M,
        "local_qwen_api_cost_usd": 0,
        "note": "Token-based estimates; billing, retries and discounts may differ.",
    }
    (assets / "experiment-rerank-candidates.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    table = pd.DataFrame(
        {
            label: {
                f"{size} candidati": report["summary"][f"{model}/{size}"]["hit_rate@5"]
                for size in SIZES
            }
            for model, label in (("gemini", "Gemini Flash"), ("qwen3", "Qwen3 0.6B"))
        }
    )
    figure = compare_chart(
        table,
        fmt="{:.1%}",
        title="Stessi 10 chunk finali: cambia la rosa dei candidati",
    )
    for suffix in ("svg", "png"):
        figure.savefig(assets / f"experiment-rerank-candidates.{suffix}", dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=227)
    parser.add_argument("--budget-usd", type=float, default=8.0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/rerank-candidates.json"
    )
    args = parser.parse_args()
    questions = [
        q
        for filename in ("faq_questions.jsonl", "image_questions.jsonl")
        for q in read_jsonl(DATA / "eval" / filename)
        if q["gold"]
    ]
    if not 1 <= args.sample <= len(questions):
        parser.error(f"--sample must be between 1 and {len(questions)}")
    questions = questions[: args.sample]
    cache_embeddings()
    cache_searches()
    cache_scores()
    cache_llm_calls()
    specification = {
        "question_ids": [q["id"] for q in questions],
        "model": CHAT_MODEL,
        "qwen": "Qwen/Qwen3-Reranker-0.6B",
        "index": "bge-m3-1000-bm25",
        "candidates": list(SIZES),
        "top_n": 10,
        "thinking_level": "low",
        "filter": "call",
    }
    report = {"specification": specification, "records": {}, "sessions": []}
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report["specification"] != specification:
            raise ValueError("Existing report has a different experiment configuration")

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temporary.replace(args.output)

    pools = {}
    store = open_store(
        ROOT / ".cache/qdrant/bge-m3-1000-bm25",
        "bge-m3-1000-bm25",
        embedding_model("bge-m3"),
        sparse_model(),
    )
    try:
        for size in SIZES:
            # RRF may change with k: fetch each size independently, never slice k=50.
            pools[size] = [
                store.similarity_search(
                    q["question"], k=size, filter=metadata_filter(call=q["call"])
                )
                for q in questions
            ]
            print(
                f"Retrieved {size} candidates for {len(questions)} questions",
                flush=True,
            )
    finally:
        store.client.close()
    del store
    gc.collect()

    qrels = gold_qrels(questions)
    report["candidate_ceiling"] = {}
    for size, pool in pools.items():
        report["candidate_ceiling"][str(size)] = sum(
            bool(set(pages_run(docs)) & set(qrels.to_dict()[q["id"]]))
            for q, docs in zip(questions, pool, strict=True)
        ) / len(questions)

    scorer = cross_encoder("qwen3-reranker-0.6b")
    for size, pool in pools.items():
        records = report["records"].setdefault(f"qwen3/{size}", {})
        for q, docs in zip(questions, pool, strict=True):
            if q["id"] in records:
                continue
            scores = scorer(q["question"], docs)
            order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
            seconds = scorer.seconds[-1]
            records[q["id"]] = {
                "pages": pages_run([docs[i] for i in order[:10]]),
                "seconds": seconds,
            }
            save()
        print(f"Qwen3 {size}: complete", flush=True)
    del scorer
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    chat = chat_model(CHAT_MODEL, thinking_level="low", timeout=60, max_retries=3)
    ranker = chat.with_structured_output(Ranking, include_raw=True)
    db = sqlite3.connect(ROOT / ".cache/llm-calls.sqlite")
    last_row = db.execute("SELECT COALESCE(MAX(rowid), 0) FROM calls").fetchone()[0]
    prior_spend = sum(s.get("estimated_usd", 0) for s in report["sessions"])
    session = {"started_utc": datetime.now(UTC).isoformat()}
    report["sessions"].append(session)
    try:
        for size, pool in pools.items():
            records = report["records"].setdefault(f"gemini/{size}", {})
            pending = [
                (q, docs)
                for q, docs in zip(questions, pool, strict=True)
                if q["id"] not in records
            ]
            for offset in range(0, len(pending), 8):
                usage = new_api_usage(db, last_row)
                if prior_spend + usage["estimated_usd"] >= args.budget_usd:
                    raise RuntimeError("API budget reached; report saved for resuming")
                batch = pending[offset : offset + 8]
                start = time.perf_counter()
                outputs = ranker.batch(
                    [listwise_prompt(q["question"], docs) for q, docs in batch],
                    config={"max_concurrency": 4},
                    return_exceptions=True,
                )
                batch_seconds = time.perf_counter() - start
                failures = []
                for (q, docs), result in zip(batch, outputs, strict=True):
                    if isinstance(result, Exception) or result["parsed"] is None:
                        failures.append(q["id"])
                        continue
                    ranking = result["parsed"].order
                    rank = {n: i for i, n in enumerate(ranking)}
                    order = sorted(
                        range(len(docs)), key=lambda i: rank.get(i + 1, len(docs))
                    )
                    records[q["id"]] = {
                        "pages": pages_run([docs[i] for i in order[:10]]),
                        "usage": result["raw"].usage_metadata,
                        "batch_wall_seconds": batch_seconds,
                        "batch_size": len(batch),
                    }
                session.update(new_api_usage(db, last_row))
                save()
                print(
                    f"Gemini {size}: {len(records)}/{len(questions)}; "
                    f"new API estimate ${session['estimated_usd']:.3f}",
                    flush=True,
                )
                if failures:
                    raise RuntimeError(
                        f"Failed questions (rerun to resume): {failures}"
                    )
    finally:
        session.update(new_api_usage(db, last_row))
        save()
        db.close()

    report["summary"] = {}
    per_question = {}
    for name, records in report["records"].items():
        run = Run({q["id"]: records[q["id"]]["pages"] for q in questions}, name=name)
        summary = dict(evaluate(qrels, run, METRICS))
        hits = [
            bool(set(records[q["id"]]["pages"]) & set(qrels.to_dict()[q["id"]]))
            for q in questions
        ]
        summary["hit_rate@10"] = float(np.mean(hits))
        if name.startswith("qwen3"):
            summary["mean_scoring_seconds"] = float(
                np.mean([record["seconds"] for record in records.values()])
            )
        else:
            summary["input_tokens"] = sum(
                r["usage"]["input_tokens"] for r in records.values()
            )
            summary["output_tokens"] = sum(
                r["usage"]["output_tokens"] for r in records.values()
            )
            summary["full_uncached_estimated_usd"] = (
                summary["input_tokens"] * INPUT_USD_PER_M
                + summary["output_tokens"] * OUTPUT_USD_PER_M
            ) / 1e6
        report["summary"][name] = summary
        per_question[name] = np.asarray(
            evaluate(qrels, run, "hit_rate@5", return_mean=False)
        )

    # Paired bootstrap: resample the same question IDs for both candidate sizes.
    rng = np.random.default_rng(0)
    indices = rng.integers(len(questions), size=(10000, len(questions)))
    report["paired_bootstrap_hit5_vs_30"] = {}
    for model in ("qwen3", "gemini"):
        for size in (20, 50):
            differences = per_question[f"{model}/{size}"] - per_question[f"{model}/30"]
            low, high = np.quantile(differences[indices].mean(axis=1), [0.025, 0.975])
            report["paired_bootstrap_hit5_vs_30"][f"{model}/{size}"] = {
                "difference": float(differences.mean()),
                "ci95": [float(low), float(high)],
            }
    save()
    write_artifacts(report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
