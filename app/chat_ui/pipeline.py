"""One question through the pipeline: search, rerank, answer, judge."""

import time
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field

import streamlit as st
from langchain_core.callbacks import (
    UsageMetadataCallbackHandler,
    get_usage_metadata_callback,
)
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from chat_ui.resources import (
    ANSWER_MODELS,
    CANDIDATES,
    GEMINI_RERANKER,
    MODES,
    NO_RERANKER,
    answer_judge,
    faq_for,
    indexes,
    llm,
    reranker,
)
from ragcourse.chat import seconds, thousands
from ragcourse.costs import api_cost
from ragcourse.filters import metadata_filter
from ragcourse.generation import PROMPT, PRUDENT_PROMPT, format_context
from ragcourse.index import search_mode
from ragcourse.judge import AnswerJudgment, judge_answer
from ragcourse.models import CHAT_MODEL, JUDGE_MODEL

Log = Callable[[str], None]  # writes one finished step in the status box


@dataclass
class Settings:
    """The levers chosen in the sidebar."""

    model: str
    judge: bool
    call: str | None
    rag: bool
    ocr: bool
    mode: str
    reranker: str
    k: int
    prompt: str
    thinking: str


@dataclass
class ApiCost:
    stage: str
    model: str
    usd: float | None  # None: the API did not report token usage


@dataclass
class Turn:
    """One question and everything shown about its answer."""

    question: str
    settings: Settings
    faq: dict | None
    steps: list[str] = field(default_factory=list)
    chunks: list[Document] = field(default_factory=list)
    positions: list[int] = field(default_factory=list)  # chunk ranks in the search
    prompt: list[BaseMessage] = field(default_factory=list)
    answer: str = ""
    api_costs: list[ApiCost] = field(default_factory=list)
    judgment: AnswerJudgment | None = None
    judge_basis: str = ""
    judge_error: bool = False


@contextmanager
def timed() -> Generator[Callable[[], str]]:
    """Yield a function returning the time elapsed since the block started."""
    start = time.perf_counter()
    yield lambda: seconds(time.perf_counter() - start)


@contextmanager
def api_call(
    turn: Turn, stage: str, model: str
) -> Generator[UsageMetadataCallbackHandler]:
    """Add to the turn the cost of the API calls made inside the block."""
    with get_usage_metadata_callback() as usage:
        yield usage
    turn.api_costs.append(ApiCost(stage, model, api_cost(model, usage.usage_metadata)))


def ask(question: str, settings: Settings, status) -> Turn:
    """Run every step, writing each one in the status box as it ends."""
    turn = Turn(question, settings, faq=faq_for(question))

    def log(text: str) -> None:
        turn.steps.append(text)
        status.markdown(text)

    if settings.rag:
        candidates = search(question, settings, log)
        turn.positions = rerank(turn, candidates, log)[: settings.k]
        turn.chunks = [candidates[i] for i in turn.positions]
    else:
        log("🚫 **Nessun documento**: il modello risponde con quello che ricorda")

    status.update(label="Scrivo la risposta…")
    answer(turn, log)
    if settings.judge:
        status.update(label="Valuto la correttezza…")
        judge(turn, log)
    return turn


def search(question: str, settings: Settings, log: Log) -> list[Document]:
    """First stage: the candidate chunks, optionally within one call."""
    store = search_mode(indexes()[settings.ocr], MODES[settings.mode])
    options = {"filter": metadata_filter(call=settings.call)} if settings.call else {}
    retriever = store.as_retriever(search_kwargs={"k": CANDIDATES})
    with timed() as elapsed:
        candidates = retriever.invoke(question, **options)
    where = "nel bando scelto" if settings.call else "in tutti i documenti"
    log(
        f"🔎 **Ricerca {settings.mode}** {where}: {len(candidates)} chunk "
        f"candidati · {elapsed()}"
    )
    return candidates


def rerank(turn: Turn, candidates: list[Document], log: Log) -> list[int]:
    """Candidate positions, best first: the search order without a reranker."""
    order = list(range(len(candidates)))
    name = turn.settings.reranker
    if name == NO_RERANKER:
        return order
    score = reranker(name)  # loaded once, not timed
    paid = api_call(turn, "Reranking", CHAT_MODEL)
    with timed() as elapsed, paid if name == GEMINI_RERANKER else nullcontext():
        scores = score(turn.question, candidates)
    order.sort(key=lambda i: scores[i], reverse=True)
    log(f"↕️ **Reranking** con {name}: riletti tutti i {len(candidates)} · {elapsed()}")
    return order


def answer(turn: Turn, log: Log) -> None:
    """Stream the answer, from the chunks or from the model's memory alone."""
    settings = turn.settings
    model = llm(settings.model, settings.thinking)
    if settings.rag:
        prompt = PRUDENT_PROMPT if settings.prompt == "prudente" else PROMPT
        inputs = {"context": format_context(turn.chunks), "question": turn.question}
        turn.prompt = prompt.invoke(inputs).to_messages()
        chain = prompt | model | StrOutputParser()
    else:
        inputs = turn.question
        turn.prompt = [HumanMessage(turn.question)]
        chain = model | StrOutputParser()

    with timed() as elapsed, api_call(turn, "Risposta", settings.model) as usage:
        turn.answer = st.empty().write_stream(chain.stream(inputs))
    log(
        f"✍️ **Risposta** di {ANSWER_MODELS[settings.model]}: "
        f"{token_summary(usage.usage_metadata)} · {elapsed()}"
    )


def token_summary(usage_metadata: dict) -> str:
    """Tokens read and written, and how many went into reasoning."""
    tokens = next(iter(usage_metadata.values()), {})
    text = (
        f"{thousands(tokens.get('input_tokens', 0))} token letti, "
        f"{thousands(tokens.get('output_tokens', 0))} scritti"
    )
    if thinking := tokens.get("output_token_details", {}).get("reasoning", 0):
        text += f" (di cui {thousands(thinking)} per ragionare)"
    return text


def judge(turn: Turn, log: Log) -> None:
    """Grade the answer against the FAQ, or else the retrieved chunks."""
    expected = turn.faq["answer"] if turn.faq else None
    has_reference = bool(expected or turn.chunks)
    if expected:
        turn.judge_basis = "Confronto con la risposta ufficiale della FAQ"
    elif turn.chunks:
        turn.judge_basis = "Verifica rispetto ai documenti recuperati"
    else:
        turn.judge_basis = "Nessun riferimento disponibile"

    paid = api_call(turn, "Giudice", JUDGE_MODEL)
    with timed() as elapsed, paid if has_reference else nullcontext():
        try:
            chat = answer_judge() if has_reference else None  # no reference: no call
            turn.judgment = judge_answer(
                chat, turn.question, turn.answer, turn.chunks, expected
            )
        except Exception:
            turn.judge_error = True
    if turn.judge_error:
        log("⚖️ **Giudizio non disponibile**: la risposta è stata conservata.")
    else:
        log(
            f"⚖️ **Giudice {ANSWER_MODELS[JUDGE_MODEL]}**: {turn.judgment.verdict}"
            f" · {elapsed()}"
        )
