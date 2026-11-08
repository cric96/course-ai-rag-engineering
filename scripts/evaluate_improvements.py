"""Evaluate the four technical improvements (1, 2, 3, 6) on the 50-FAQ sample.

1. Grounded correctness + taxonomy of FAQ support (explicit / deductive / external)
2. Question-aware context coverage metric vs DeepEval ContextualRecallMetric
3. Document titles/pages passed to the reranker and page headers
6. Deductive prompt allowing logical conclusions from exhaustive rules
"""

import argparse
import json
import sqlite3
from pathlib import Path

import ragcourse  # noqa: F401  # disable DeepEval telemetry before importing it

# isort: split

import pandas as pd
from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.test_case import LLMTestCase
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from ragcourse.cache import cache_llm_calls
from ragcourse.corpus import load_pages, with_headers
from ragcourse.evaluation import gold_found
from ragcourse.filters import metadata_filter
from ragcourse.generation import (
    ABSTAIN,
    DEDUCTIVE_PROMPT,
    expand_pages,
    format_context,
)
from ragcourse.io import read_jsonl
from ragcourse.judge import (
    LangChainJudge,
    context_coverage_metric,
    correctness_metric,
    deepeval_table,
    grounded_correctness_metric,
)
from ragcourse.models import CHAT_MODEL, JUDGE_MODEL, chat_model
from ragcourse.paths import DATA, ROOT
from ragcourse.rerank import Ranking, listwise_prompt
from ragcourse.viz import bar_chart, compare_chart

# FAQ where the official answer quotes external laws, errata corrige, or UI rules.
EXTERNAL_FAQS = {
    "faq-051",
    "faq-052",
    "faq-067",
    "faq-107",
    "faq-113",
    "faq-180",
    "faq-182",
    "faq-189",
    "faq-224",
}

# FAQ where the call states a general rule and the FAQ deduces a specific case.
DEDUCTIVE_FAQS = {
    "faq-006",
    "faq-014",
    "faq-017",
    "faq-020",
    "faq-025",
    "faq-041",
    "faq-082",
    "faq-100",
    "faq-138",
    "faq-144",
    "faq-146",
    "faq-148",
    "faq-184",
}


def classify_questions(question_ids: list[str]) -> dict[str, list[str]]:
    """Partition question IDs into explicit, deductive, and external-source groups."""
    groups = {"explicit": [], "deductive": [], "external": []}
    for qid in question_ids:
        if qid in EXTERNAL_FAQS:
            groups["external"].append(qid)
        elif qid in DEDUCTIVE_FAQS:
            groups["deductive"].append(qid)
        else:
            groups["explicit"].append(qid)
    return groups


def context_text(docs: list[Document]) -> str:
    """Format retrieved chunks/pages with their numbered citations."""
    return format_context(docs)


def load_hybrid_candidates(
    questions: list[dict], search_cache: Path
) -> list[list[Document]]:
    """Read the 30 hybrid candidates per question from the search cache."""
    candidates = []
    with sqlite3.connect(f"file:{search_cache.resolve()}?mode=ro", uri=True) as db:
        for q in questions:
            key = json.dumps(
                [
                    "qdrant:bge-m3-1000-bm25",
                    "hybrid",
                    q["question"],
                    30,
                    repr(metadata_filter(call=q["call"])),
                ]
            )
            row = db.execute(
                "SELECT results FROM searches WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Missing hybrid candidates in cache: {q['id']}")
            docs = [
                Document(page_content=text, metadata=metadata, id=doc_id)
                for text, metadata, doc_id, _ in json.loads(row[0])
            ]
            candidates.append(docs)
    return candidates


def rerank_with_titles(
    questions: list[dict], candidates: list[list[Document]], top_n: int = 10
) -> list[list[Document]]:
    """Rerank the 30 candidates using Gemini listwise with document titles and pages."""
    chat = chat_model(CHAT_MODEL, thinking_level="low", timeout=60, max_retries=3)
    prompts = [
        listwise_prompt(q["question"], docs, with_titles=True)
        for q, docs in zip(questions, candidates, strict=True)
    ]
    orders = chat.with_structured_output(Ranking).batch(
        prompts, config={"max_concurrency": 8}
    )
    selected = []
    for docs, order in zip(candidates, orders, strict=True):
        rank = {n: i for i, n in enumerate(order.order)}
        scores = [
            float(len(docs) - rank.get(n, len(docs))) for n in range(1, len(docs) + 1)
        ]
        best = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        selected.append([docs[i] for i in best[:top_n]])
    return selected


def evaluate_context_coverage(judge) -> dict:
    """Compare DeepEval ContextualRecallMetric with question-aware GEval coverage."""
    cr_report = json.loads((ROOT / ".cache/context-recall-k.json").read_text())
    faq_all = {q["id"]: q for q in read_jsonl(DATA / "eval/faq_questions.jsonl")}
    quiet = DisplayConfig(
        print_results=False, show_indicator=False, inspect_after_run=False
    )
    metric = context_coverage_metric(judge)
    cases = []
    for k in ["5", "10", "20"]:
        for record in cr_report["variants"][k]:
            cases.append(
                LLMTestCase(
                    name=f"{k}/{record['id']}",
                    input=record["question"],
                    expected_output=record["expected"],
                    retrieval_context=[c["text"] for c in record["chunks"]],
                )
            )
    result = evaluate(
        cases,
        [metric],
        async_config=AsyncConfig(max_concurrent=10),
        display_config=quiet,
    )
    scores = deepeval_table(result).iloc[:, 0]
    summary = {}
    for k in ["5", "10", "20"]:
        recs = cr_report["variants"][k]
        vals = [float(scores[f"{k}/{r['id']}"]) for r in recs]
        with_gold, without_gold = [], []
        rec_with_gold, rec_without_gold = [], []
        for r, v in zip(recs, vals, strict=True):
            gold = {(g["url"], g["page"]) for g in faq_all[r["id"]]["gold"]}
            found = {(c["metadata"]["url"], c["metadata"]["page"]) for c in r["chunks"]}
            if gold & found:
                with_gold.append(v)
                rec_with_gold.append(r["score"])
            else:
                without_gold.append(v)
                rec_without_gold.append(r["score"])
        summary[k] = {
            "contextual_recall": cr_report["summary"][k]["mean"],
            "context_coverage_geval": sum(vals) / len(vals),
            "zero_contextual_recall": cr_report["summary"][k]["zero"],
            "zero_context_coverage": sum(v == 0 for v in vals),
            "by_gold": {
                "absent": {
                    "contextual_recall": sum(rec_without_gold) / len(rec_without_gold),
                    "context_coverage_geval": sum(without_gold) / len(without_gold),
                    "count": len(without_gold),
                },
                "present": {
                    "contextual_recall": sum(rec_with_gold) / len(rec_with_gold),
                    "context_coverage_geval": sum(with_gold) / len(with_gold),
                    "count": len(with_gold),
                },
            },
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/improvements.json"
    )
    parser.add_argument("--assets", type=Path, default=ROOT / "slides/assets")
    args = parser.parse_args()

    cache_llm_calls()
    questions = (
        pd.DataFrame(
            [q for q in read_jsonl(DATA / "eval/faq_questions.jsonl") if q["gold"]]
        )
        .sample(50, random_state=0)
        .to_dict("records")
    )
    qids = [q["id"] for q in questions]
    groups = classify_questions(qids)

    judge = LangChainJudge(
        chat_model(JUDGE_MODEL, timeout=120, max_retries=3), JUDGE_MODEL
    )
    print("1/4 Evaluating question-aware context coverage (k=5, 10, 20)...", flush=True)
    coverage_summary = evaluate_context_coverage(judge)
    print(json.dumps(coverage_summary, indent=2), flush=True)

    print("2/4 Reranking 30 hybrid candidates with document titles...", flush=True)
    candidates = load_hybrid_candidates(questions, ROOT / ".cache/searches.sqlite")
    selected_chunks = rerank_with_titles(questions, candidates, top_n=10)

    pages = with_headers(load_pages())
    by_page = {(p.metadata["url"], p.metadata["page"]): p for p in pages}
    selected_pages = [expand_pages(docs, by_page) for docs in selected_chunks]

    # Load baseline page-context results (Flash and Argon).
    base_flash = json.loads((ROOT / ".cache/page-context.json").read_text())
    base_argon = json.loads((ROOT / ".cache/page-context-argon.json").read_text())

    print(
        "3/4 Generating answers with DEDUCTIVE_PROMPT (Flash and Argon)...", flush=True
    )
    runs = {}
    for model_label, model_name in [
        ("flash", CHAT_MODEL),
        ("argon", "google_genai:gemini-4-argon"),
    ]:
        llm = chat_model(model_name, timeout=120, max_retries=3)
        chain = DEDUCTIVE_PROMPT | llm | StrOutputParser()
        for ctx_label, selections in [
            ("chunk", selected_chunks),
            ("pagine", selected_pages),
        ]:
            inputs = [
                {"context": context_text(docs), "question": q["question"]}
                for q, docs in zip(questions, selections, strict=True)
            ]
            answers = [None] * len(inputs)
            for done, (idx, ans) in enumerate(
                chain.batch_as_completed(inputs, config={"max_concurrency": 6}), start=1
            ):
                answers[idx] = ans
                if done % 10 == 0 or done == len(inputs):
                    print(
                        f"  {model_label}_{ctx_label}: {done}/{len(inputs)}",
                        flush=True,
                    )
            key = f"{model_label}_{ctx_label}_deductive"
            runs[key] = [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "expected": q["answer"],
                    "answer": ans,
                    "abstained": ans.startswith(ABSTAIN),
                    "gold_found": gold_found(docs, q),
                    "context": [d.page_content for d in docs],
                }
                for q, docs, ans in zip(questions, selections, answers, strict=True)
            ]
            print(f"Generated {key}", flush=True)

    # Build test cases for baseline Flash/Argon pagine to judge grounded correctness.
    raw_pages = {(p.metadata["url"], p.metadata["page"]): p for p in load_pages()}
    base_candidates = [
        [
            raw_pages[(u.split("#page=")[0], int(u.split("#page=")[1]))]
            for u in rec["pages"]
        ]
        for rec in base_flash["variants"]["pagine"]
    ]
    for model_label, base_report in [("flash", base_flash), ("argon", base_argon)]:
        key = f"{model_label}_pagine_baseline"
        runs[key] = [
            {
                "id": rec["id"],
                "question": rec["question"],
                "expected": rec["expected"],
                "answer": rec["answer"],
                "abstained": rec["abstained"],
                "gold_found": rec["gold_found"],
                "context": [d.page_content for d in docs],
                "std_score": rec["score"],
            }
            for rec, docs in zip(
                base_report["variants"]["pagine"], base_candidates, strict=True
            )
        ]

    print("4/4 Judging standard and grounded correctness...", flush=True)
    quiet = DisplayConfig(
        print_results=False, show_indicator=False, inspect_after_run=False
    )
    std_cases, grd_cases = [], []
    for run_key, records in runs.items():
        for rec in records:
            case = LLMTestCase(
                name=f"{run_key}/{rec['id']}",
                input=rec["question"],
                actual_output=rec["answer"],
                expected_output=rec["expected"],
                retrieval_context=rec["context"],
            )
            if "std_score" not in rec:
                std_cases.append(case)
            grd_cases.append(case)

    std_res = evaluate(
        std_cases,
        [correctness_metric(judge)],
        async_config=AsyncConfig(max_concurrent=10),
        display_config=quiet,
    )
    grd_res = evaluate(
        grd_cases,
        [grounded_correctness_metric(judge)],
        async_config=AsyncConfig(max_concurrent=10),
        display_config=quiet,
    )
    std_scores = deepeval_table(std_res).iloc[:, 0]
    grd_scores = deepeval_table(grd_res).iloc[:, 0]

    summary_runs = {}
    for run_key, records in runs.items():
        for rec in records:
            name = f"{run_key}/{rec['id']}"
            if "std_score" not in rec:
                rec["std_score"] = float(std_scores[name])
            rec["grd_score"] = float(grd_scores[name])
        by_id = {r["id"]: r for r in records}
        summary_runs[run_key] = {
            "correctness": sum(r["std_score"] for r in records) / len(records),
            "grounded_correctness": sum(r["grd_score"] for r in records) / len(records),
            "abstained": sum(r["abstained"] for r in records),
            "gold_found": sum(r["gold_found"] for r in records),
            "by_group_correctness": {
                g: sum(by_id[i]["std_score"] for i in ids) / len(ids)
                for g, ids in groups.items()
            },
            "by_group_grounded": {
                g: sum(by_id[i]["grd_score"] for i in ids) / len(ids)
                for g, ids in groups.items()
            },
        }

    report = {
        "questions": 50,
        "groups": {k: {"count": len(v), "ids": v} for k, v in groups.items()},
        "context_coverage": coverage_summary,
        "runs": summary_runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    args.assets.mkdir(parents=True, exist_ok=True)
    (args.assets / "experiment-improvements.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    # Chart 1: Context Recall (DeepEval) vs Context Coverage (GEval) in chapter 04 (k=5)
    k5 = coverage_summary["5"]
    cov_ch04 = pd.DataFrame(
        {
            "Contextual Recall (solo risposta)": {
                "tutte le 50 FAQ": k5["contextual_recall"],
                "pagina giusta assente": k5["by_gold"]["absent"]["contextual_recall"],
                "pagina giusta nei 5 chunk": k5["by_gold"]["present"][
                    "contextual_recall"
                ],
            },
            "Copertura GEval (domanda + risposta)": {
                "tutte le 50 FAQ": k5["context_coverage_geval"],
                "pagina giusta assente": k5["by_gold"]["absent"][
                    "context_coverage_geval"
                ],
                "pagina giusta nei 5 chunk": k5["by_gold"]["present"][
                    "context_coverage_geval"
                ],
            },
        }
    )
    chart_ch04 = compare_chart(
        cov_ch04,
        fmt="{:.2f}",
        title="Stessi 5 chunk: Contextual Recall contro Copertura con la domanda",
    )
    chart_ch04.set_size_inches(12, 4.5)
    for ext in [".svg", ".png"]:
        chart_ch04.savefig(
            args.assets / f"experiment-context-coverage-ch04{ext}",
            bbox_inches="tight",
            dpi=150,
        )

    cov_df = pd.DataFrame(
        {
            "Contextual Recall (solo risposta)": {
                f"{k} chunk": coverage_summary[k]["contextual_recall"]
                for k in ["5", "10", "20"]
            },
            "Copertura GEval (domanda + risposta)": {
                f"{k} chunk": coverage_summary[k]["context_coverage_geval"]
                for k in ["5", "10", "20"]
            },
        }
    )
    chart1 = compare_chart(
        cov_df,
        fmt="{:.2f}",
        title="Richiamo letterale della FAQ contro copertura reale del contesto",
    )
    chart1.set_size_inches(11.5, 4.2)
    for ext in [".svg", ".png"]:
        chart1.savefig(
            args.assets / f"experiment-context-coverage{ext}",
            bbox_inches="tight",
            dpi=150,
        )

    # Chart 2: Progression of correctness (standard vs grounded, base vs deductive)
    prog = pd.Series(
        {
            "Flash · pagine (base)": summary_runs["flash_pagine_baseline"][
                "correctness"
            ],
            "Flash · + titoli e prompt deduttivo": summary_runs[
                "flash_pagine_deductive"
            ]["correctness"],
            "Argon · pagine (base)": summary_runs["argon_pagine_baseline"][
                "correctness"
            ],
            "Argon · + titoli e prompt deduttivo": summary_runs[
                "argon_pagine_deductive"
            ]["correctness"],
            "Argon · sulle FAQ esplicite nei doc": summary_runs[
                "argon_pagine_deductive"
            ]["by_group_correctness"]["explicit"],
            "Argon · correttezza nel contesto": summary_runs["argon_pagine_deductive"][
                "grounded_correctness"
            ],
        }
    )
    chart2 = bar_chart(
        prog,
        gray=["Flash · pagine (base)"],
        fmt="{:.2f}",
        title="Dalla correttezza grezza (0,54) alla correttezza rispetto ai documenti",
    )
    chart2.set_size_inches(12, 4.6)
    for ext in [".svg", ".png"]:
        chart2.savefig(
            args.assets / f"experiment-improvements{ext}",
            bbox_inches="tight",
            dpi=150,
        )

    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
