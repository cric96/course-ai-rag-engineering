"""Fixed RAG pipeline vs a LangChain RAG agent, on the 50 FAQ of the generation deck.

Same retriever (bge-m3 + BM25 hybrid, call filter, 10 chunks), same generator and judge
(Gemini Flash). The agent decides when and what to search; the pipeline searches once.

    uv run python scripts/compare_agent.py
"""

import json

import pandas as pd
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig, DisplayConfig
from deepeval.test_case import LLMTestCase
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragcourse.cache import cache_embeddings, cache_llm_calls, cache_searches
from ragcourse.corpus import load_pages
from ragcourse.filters import metadata_filter
from ragcourse.generation import ABSTAIN, PARTIAL_OK, PROMPT, format_context
from ragcourse.index import vector_store
from ragcourse.io import read_jsonl
from ragcourse.judge import correctness_metric, deepeval_table, judge_model
from ragcourse.models import JUDGE_MODEL, chat_model, embedding_model, sparse_model
from ragcourse.paths import DATA, ROOT

K = 10
OUTPUT = ROOT / ".cache/agent-comparison.json"

# The pipeline's rules; citations name title and page, since the agent may search twice.
AGENT_RULES = (
    "Rispondi a domande sui bandi del PR FESR 2021-2027 dell'Emilia-Romagna.\n"
    "Cerca nei documenti con lo strumento cerca_documenti: puoi cercare più volte, "
    "con parole diverse, se i primi risultati non bastano.\n"
    "Usa solo i documenti trovati. Rispondi in modo preciso, con le condizioni\n"
    "e le eccezioni che contano. Cita le fonti come (titolo, p. N).\n" + PARTIAL_OK
)
PASSAGE = PromptTemplate.from_template("[{title}, p. {page}]\n{page_content}")


def ask_agent(llm, store, question: dict) -> dict:
    """One agent per question: its search tool is bound to the question's call."""
    retriever = store.as_retriever(
        search_kwargs={"k": K, "filter": metadata_filter(call=question["call"])}
    )
    tool = create_retriever_tool(
        retriever,
        name="cerca_documenti",
        description="Cerca passaggi nei documenti del bando: domanda o parole chiave.",
        document_prompt=PASSAGE,
        response_format="content_and_artifact",
    )
    agent = create_agent(llm, tools=[tool], system_prompt=AGENT_RULES)
    state = agent.invoke({"messages": [("user", question["question"])]})
    searches = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    chunks = [doc for m in searches for doc in (m.artifact or [])]
    return {
        "answer": state["messages"][-1].text,
        # The queries the agent chose, in order.
        "searches": [
            call["args"].get("query", "")
            for m in state["messages"]
            for call in getattr(m, "tool_calls", [])
        ],
        "context": [doc.page_content for doc in chunks],
    }


def main() -> None:
    cache_llm_calls()
    cache_embeddings()
    cache_searches()
    questions = (
        pd.DataFrame(
            [q for q in read_jsonl(DATA / "eval/faq_questions.jsonl") if q["gold"]]
        )
        .sample(50, random_state=0)
        .to_dict("records")
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    store = vector_store(
        splitter.split_documents(load_pages()),
        embedding_model("bge-m3"),
        "bge-m3-1000-bm25",
        sparse=sparse_model("bm25"),
    )
    llm = chat_model()

    # Fixed pipeline: one search, then the course prompt.
    pipeline = []
    for q in questions:
        docs = store.as_retriever(search_kwargs={"k": K}).invoke(
            q["question"], filter=metadata_filter(call=q["call"])
        )
        answer = (PROMPT | llm | StrOutputParser()).invoke(
            {"context": format_context(docs), "question": q["question"]}
        )
        pipeline.append(
            {
                "answer": answer,
                "searches": [q["question"]],
                "context": [d.page_content for d in docs],
            }
        )
        print(f"pipeline {len(pipeline)}/50", flush=True)

    agent = []
    for q in questions:
        agent.append(ask_agent(llm, store, q))
        print(
            f"agent {len(agent)}/50: {len(agent[-1]['searches'])} searches", flush=True
        )

    cases = [
        LLMTestCase(
            name=f"{label}/{q['id']}",
            input=q["question"],
            actual_output=run["answer"] or "(vuota)",
            expected_output=q["answer"],
            retrieval_context=run["context"] or [""],
        )
        for label, runs in [("pipeline", pipeline), ("agent", agent)]
        for q, run in zip(questions, runs, strict=True)
    ]
    result = evaluate(
        cases,
        [correctness_metric(judge_model(JUDGE_MODEL))],
        async_config=AsyncConfig(max_concurrent=10),
        display_config=DisplayConfig(print_results=False, show_indicator=False),
    )
    scores = deepeval_table(result).iloc[:, 0]

    report = {}
    for label, runs in [("pipeline", pipeline), ("agent", agent)]:
        per_q = [float(scores[f"{label}/{q['id']}"]) for q in questions]
        for run, score in zip(runs, per_q, strict=True):
            run["correctness"] = score
        report[label] = {
            "correctness": sum(per_q) / len(per_q),
            "abstained": sum(r["answer"].startswith(ABSTAIN) for r in runs),
            "searches_per_question": sum(len(r["searches"]) for r in runs) / len(runs),
            "records": [
                {"id": q["id"], **r} for q, r in zip(questions, runs, strict=True)
            ],
        }
    delta = pd.Series(
        [
            a["correctness"] - p["correctness"]
            for a, p in zip(agent, pipeline, strict=True)
        ]
    )
    report["delta"] = {
        "mean": float(delta.mean()),
        "better": int((delta > 0).sum()),
        "same": int((delta == 0).sum()),
        "worse": int((delta < 0).sum()),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    for label in ["pipeline", "agent"]:
        r = report[label]
        print(
            f"{label}: correctness {r['correctness']:.3f}, abstained {r['abstained']}, "
            f"searches/question {r['searches_per_question']:.1f}"
        )
    print("agent - pipeline:", report["delta"])


if __name__ == "__main__":
    main()
