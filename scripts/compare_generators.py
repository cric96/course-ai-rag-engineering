"""Compare generators with the generation chapter's fixed retrieval and judge.

uv run python scripts/compare_generators.py --model google_genai:<model>
"""

import argparse
import json
from pathlib import Path

import ragcourse  # noqa: F401  # disable DeepEval telemetry before importing it

# isort: split

import numpy as np
import pandas as pd
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragcourse.cache import cache_embeddings, cache_llm_calls, cache_searches
from ragcourse.corpus import load_pages
from ragcourse.filters import metadata_filter
from ragcourse.generation import ABSTAIN, PROMPT, format_context
from ragcourse.index import vector_store
from ragcourse.io import read_jsonl
from ragcourse.judge import correctness_metric, deepeval_table, judge_model
from ragcourse.models import CHAT_MODEL, JUDGE_MODEL, chat_model, embedding_model
from ragcourse.paths import DATA, ROOT
from ragcourse.viz import compare_chart

METRICS = {
    "Correttezza [GEval]": "correttezza",
    "Faithfulness": "fedeltà",
    "Answer Relevancy": "pertinenza",
    "Contextual Recall": "richiamo del contesto",
}


def common_answered_metrics(runs: dict) -> dict:
    """Compare fidelity and relevancy on questions answered by both generators."""
    indexed = {
        label: {record["id"]: record for record in run["records"]}
        for label, run in runs.items()
    }
    common = sorted(
        set.intersection(
            *(
                {key for key, record in records.items() if not record["abstained"]}
                for records in indexed.values()
            )
        )
    )
    return {
        "count": len(common),
        "ids": common,
        "means": {
            label: {
                metric: sum(records[key]["scores"][metric] for key in common)
                / len(common)
                if common
                else None
                for metric in ["fedeltà", "pertinenza"]
            }
            for label, records in indexed.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--baseline", default=CHAT_MODEL)
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/generator-comparison.json"
    )
    args = parser.parse_args()
    if not 1 <= args.sample <= 50:
        parser.error("--sample must be between 1 and 50")
    if args.model == args.baseline:
        parser.error("--model must differ from --baseline")

    cache_llm_calls()
    # Fail before retrieval when the candidate model is unavailable.
    candidate = chat_model(args.model, timeout=120, max_retries=2)
    candidate.invoke("Rispondi solo con: OK")
    cache_embeddings()
    cache_searches()
    questions = (
        pd.DataFrame(
            [q for q in read_jsonl(DATA / "eval/faq_questions.jsonl") if q["gold"]]
        )
        .sample(50, random_state=0)
        .head(args.sample)
        .to_dict("records")
    )
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(load_pages())
    store = vector_store(chunks, embedding_model("e5-small"), "e5-small-1000")
    retriever = store.as_retriever(search_kwargs={"k": 5})
    found = []
    try:
        for q in questions:
            found.append(
                retriever.invoke(q["question"], filter=metadata_filter(call=q["call"]))
            )
            print(f"Retrieved {len(found)}/{len(questions)}", flush=True)
    finally:
        store.client.close()
    inputs = [
        {"context": format_context(docs), "question": q["question"]}
        for q, docs in zip(questions, found, strict=True)
    ]
    report = {
        "sample_seed": 0,
        "judge": JUDGE_MODEL,
        "retrieval": (
            "e5-small; 1000 characters, overlap 200; call filter; top 5; no reranking"
        ),
        "prompt": PROMPT.format(context="{context}", question="{question}"),
        "questions": [
            {"id": q["id"], "expected": q["answer"], **item}
            for q, item in zip(questions, inputs, strict=True)
        ],
        "runs": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    cases = []
    for label, model in [("baseline", args.baseline), ("candidate", args.model)]:
        generator = candidate if label == "candidate" else chat_model(args.baseline)
        chain = PROMPT | generator | StrOutputParser()
        records = [{"id": q["id"]} for q in questions]
        report["runs"][label] = {"model": model, "records": records}
        for done, (index, answer) in enumerate(
            chain.batch_as_completed(inputs, config={"max_concurrency": 4}), start=1
        ):
            records[index].update(answer=answer, abstained=answer.startswith(ABSTAIN))
            save()
            print(f"{label}: generated {done}/{len(questions)}", flush=True)
        for q, docs, record in zip(questions, found, records, strict=True):
            cases.append(
                LLMTestCase(
                    name=f"{label}/{q['id']}",
                    input=q["question"],
                    actual_output=record["answer"],
                    expected_output=q["answer"],
                    retrieval_context=[doc.page_content for doc in docs],
                )
            )

    judge = judge_model(JUDGE_MODEL)
    print("Judging both generators with the same four metrics", flush=True)
    result = evaluate(
        cases,
        [
            correctness_metric(judge),
            FaithfulnessMetric(model=judge, penalize_ambiguous_claims=True),
            AnswerRelevancyMetric(model=judge),
            ContextualRecallMetric(model=judge),
        ],
        async_config=AsyncConfig(max_concurrent=10),
        display_config=DisplayConfig(
            print_results=False, show_indicator=False, inspect_after_run=False
        ),
    )
    scores = deepeval_table(result).rename(columns=METRICS)
    reasons = deepeval_table(result, field="reason").rename(columns=METRICS)
    means = {}
    for label, run in report["runs"].items():
        for record in run["records"]:
            name = f"{label}/{record['id']}"
            record["scores"] = {m: float(scores.loc[name, m]) for m in METRICS.values()}
            record["reasons"] = {m: reasons.loc[name, m] for m in METRICS.values()}
        frame = pd.DataFrame([r["scores"] for r in run["records"]])
        given = [not r["abstained"] for r in run["records"]]
        # Match the chapter: fidelity and relevancy only describe real answers.
        frame.loc[[not value for value in given], ["fedeltà", "pertinenza"]] = np.nan
        run["means"] = frame.mean().to_dict()
        run["answers_given"] = sum(given)
        means[run["model"]] = run["means"]
    paired = pd.DataFrame(
        {
            label: {r["id"]: r["scores"]["correttezza"] for r in run["records"]}
            for label, run in report["runs"].items()
        }
    )
    delta = paired["candidate"] - paired["baseline"]
    report["comparison"] = {
        "correctness_delta": float(delta.mean()),
        "paired_bootstrap_95": np.quantile(
            np.random.default_rng(0)
            .choice(delta.to_numpy(), (10000, len(delta)))
            .mean(axis=1),
            [0.025, 0.975],
        ).tolist(),
        "improved": int((delta > 0).sum()),
        "unchanged": int((delta == 0).sum()),
        "worse": int((delta < 0).sum()),
    }
    report["common_answered"] = common_answered_metrics(report["runs"])
    save()
    table = pd.DataFrame(means)
    for label, values in report["common_answered"]["means"].items():
        for metric, value in values.items():
            table.loc[metric, report["runs"][label]["model"]] = value
    labels = {
        "google_genai:gemini-3.8-flash": "Gemini 3.8 Flash",
        "google_genai:gemini-4-argon": "Gemini 4 Argon",
        "google_genai:gemma-4-31b-it": "Gemma 4 31B · open",
    }
    chart = compare_chart(
        table.rename(columns=labels),
        fmt="{:.2f}",
        title=f"Stessi 5 chunk, due generatori · {len(questions)} FAQ",
    )
    chart.set_size_inches(12, 4.2)
    for suffix in [".svg", ".png"]:
        chart.savefig(args.output.with_suffix(suffix), bbox_inches="tight", dpi=150)
    print(table.to_string(), flush=True)
    print(json.dumps(report["comparison"], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
