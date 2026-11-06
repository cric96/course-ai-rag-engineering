"""Compare selected chunks with their full pages on the course's 50 FAQ.

    uv run python scripts/compare_page_context.py
    uv run python scripts/compare_page_context.py --sample 5

Default retrieval: bge-m3 + BM25, call filter, Gemini reranking, best 10 of 30.
Gold pages are used only for reporting retrieval hits, never to build the context.
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
from deepeval.test_case import LLMTestCase
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragcourse.cache import (
    cache_embeddings,
    cache_llm_calls,
    cache_scores,
    cache_searches,
)
from ragcourse.corpus import load_pages
from ragcourse.evaluation import gold_found
from ragcourse.filters import metadata_filter
from ragcourse.generation import ABSTAIN, PROMPT, expand_pages, format_context
from ragcourse.index import vector_store
from ragcourse.io import read_jsonl
from ragcourse.judge import correctness_metric, deepeval_table, judge_model
from ragcourse.models import (
    CHAT_MODEL,
    JUDGE_MODEL,
    chat_model,
    embedding_model,
    sparse_model,
)
from ragcourse.paths import DATA, ROOT
from ragcourse.rerank import RerankRetriever, cross_encoder, llm_listwise
from ragcourse.viz import bar_chart, compare_chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=50)
    parser.add_argument("--model", default=CHAT_MODEL)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--local-threads", type=int, default=1)
    parser.add_argument("--reranker", choices=["gemini", "bge"], default="gemini")
    parser.add_argument(
        "--context", choices=["chunk", "pagine", "entrambi"], default="entrambi"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/page-context.json"
    )
    args = parser.parse_args()
    if not 1 <= args.sample <= 50:
        parser.error("--sample must be between 1 and 50")
    if args.load_in_4bit and not args.model.startswith("local:"):
        parser.error("--load-in-4bit requires a local: model")

    cache_embeddings()
    cache_llm_calls()
    cache_scores()
    cache_searches()
    # Sample 50 first: smaller smoke tests remain a subset of the course sample.
    questions = (
        pd.DataFrame(
            [q for q in read_jsonl(DATA / "eval/faq_questions.jsonl") if q["gold"]]
        )
        .sample(50, random_state=0)
        .head(args.sample)
        .to_dict("records")
    )
    pages = load_pages()
    by_page = {(p.metadata["url"], p.metadata["page"]): p for p in pages}
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(pages)
    store = vector_store(
        chunks, embedding_model("bge-m3"), "bge-m3-1000-bm25", sparse=sparse_model()
    )
    scorer = (
        llm_listwise(
            chat_model(CHAT_MODEL, thinking_level="low", timeout=60, max_retries=3)
        )
        if args.reranker == "gemini"
        else cross_encoder("bge-reranker-v2-m3")
    )
    retriever = RerankRetriever(
        base=store.as_retriever(search_kwargs={"k": 30}), scorer=scorer, top_n=10
    )
    found = []
    try:
        for q in questions:
            found.append(
                retriever.invoke(q["question"], filter=metadata_filter(call=q["call"]))
            )
            print(f"Retrieved {len(found)}/{len(questions)}", flush=True)
    finally:
        store.client.close()
    contexts = {
        "chunk": found,
        "pagine": [expand_pages(docs, by_page) for docs in found],
    }
    if args.context != "entrambi":
        contexts = {args.context: contexts[args.context]}
    # Free GPU models before loading a local generator.
    del retriever, scorer, store, chunks
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    if args.model.startswith("local:"):
        torch.set_num_threads(args.local_threads)

    settings = (
        {"load_in_4bit": args.load_in_4bit} if args.model.startswith("local:") else {}
    )
    generator = chat_model(args.model, **settings)
    chain = PROMPT | generator | StrOutputParser()
    report = {
        "model": args.model,
        "generation_settings": settings,
        "local_output_limits": [1024, 2048]
        if args.model.startswith("local:")
        else None,
        "judge": JUDGE_MODEL,
        "reranker": args.reranker,
        "sample_seed": 0,
        "retrieval": "bge-m3 + Italian BM25; call filter; best 10 of 30; no OCR",
        "variants": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))

    cases = []
    for variant, selections in contexts.items():
        print(f"Generating: {variant}", flush=True)
        inputs = [
            {"context": format_context(docs), "question": q["question"]}
            for q, docs in zip(questions, selections, strict=True)
        ]
        if args.model.startswith("local:"):
            answers = []
            for item in inputs:
                answer = chain.invoke(item)
                tokens = generator.tokenizer.encode(answer, add_special_tokens=False)
                if len(tokens) >= 1024:
                    print("Output at limit: retrying with 2048 new tokens", flush=True)
                    longer = (
                        PROMPT
                        | generator.bind(pipeline_kwargs={"max_new_tokens": 2048})
                        | StrOutputParser()
                    )
                    answer = longer.invoke(item)
                answers.append(answer)
                print(f"Generated {len(answers)}/{len(inputs)}", flush=True)
        else:
            answers = chain.batch(inputs, config={"max_concurrency": 8})
        records = []
        for q, docs, answer, inputs_q in zip(
            questions, selections, answers, inputs, strict=True
        ):
            records.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "expected": q["answer"],
                    "answer": answer,
                    "output_tokens": (
                        len(
                            generator.tokenizer.encode(answer, add_special_tokens=False)
                        )
                        if args.model.startswith("local:")
                        else None
                    ),
                    "abstained": answer.startswith(ABSTAIN),
                    "gold_found": gold_found(docs, q),
                    "context_chars": len(inputs_q["context"]),
                    "pages": list(
                        dict.fromkeys(
                            f"{d.metadata['url']}#page={d.metadata['page']}"
                            for d in docs
                        )
                    ),
                }
            )
            cases.append(
                LLMTestCase(
                    name=f"{variant}/{q['id']}",
                    input=q["question"],
                    actual_output=answer,
                    expected_output=q["answer"],
                )
            )
        report["variants"][variant] = records
        save()  # answers survive an interrupted judge run

    print("Judging with the common model and rubric", flush=True)
    result = evaluate(
        cases,
        [correctness_metric(judge_model(JUDGE_MODEL))],
        async_config=AsyncConfig(max_concurrent=10),
        display_config=DisplayConfig(
            print_results=False, show_indicator=False, inspect_after_run=False
        ),
    )
    scores = deepeval_table(result).iloc[:, 0]
    reasons = deepeval_table(result, field="reason").iloc[:, 0]
    for variant, records in report["variants"].items():
        for record in records:
            name = f"{variant}/{record['id']}"
            record.update(score=float(scores[name]), reason=reasons[name])
    paired = pd.DataFrame(
        {
            variant: {r["id"]: r["score"] for r in records}
            for variant, records in report["variants"].items()
        }
    )
    report["summary"] = {
        "correctness": paired.mean().to_dict(),
    }
    if len(paired.columns) == 2:
        delta = paired["pagine"] - paired["chunk"]
        report["summary"].update(
            mean_delta=float(delta.mean()),
            improved=int((delta > 0).sum()),
            unchanged=int((delta == 0).sum()),
            worse=int((delta < 0).sum()),
            paired_bootstrap_95=np.quantile(
                np.random.default_rng(0)
                .choice(delta.to_numpy(), (10000, len(delta)))
                .mean(axis=1),
                [0.025, 0.975],
            ).tolist(),
        )
    save()
    title = "Stesso retrieval: chunk o pagine complete al generatore?"
    chart = (
        compare_chart(
            paired.mean().to_frame("correttezza").T, fmt="{:.2f}", title=title
        )
        if len(paired.columns) == 2
        else bar_chart(paired.mean(), fmt="{:.2f}", title="Correttezza delle risposte")
    )
    chart.savefig(args.output.with_suffix(".svg"), bbox_inches="tight")
    chart.savefig(args.output.with_suffix(".png"), bbox_inches="tight", dpi=150)
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
