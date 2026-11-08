"""The course chat: the pipeline built in the slides, showing what happens behind it.

Run from the project root:  uv run streamlit run app/chat.py
"""

import time

import streamlit as st
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragcourse.chat import call_label, cited, link_citations
from ragcourse.corpus import load_pages
from ragcourse.costs import PRICING_CHECKED, PRICING_URL, api_cost
from ragcourse.evaluation import gold_found, page_id
from ragcourse.filters import metadata_filter
from ragcourse.generation import ABSTAIN, PROMPT, PRUDENT_PROMPT, format_context
from ragcourse.index import extended_store, search_mode, vector_store
from ragcourse.io import read_jsonl
from ragcourse.judge import judge_answer
from ragcourse.models import (
    CHAT_MODEL,
    JUDGE_MODEL,
    LABELING_MODEL,
    chat_model,
    embedding_model,
    sparse_model,
)
from ragcourse.ocr import page_picture
from ragcourse.paths import DATA
from ragcourse.rerank import cross_encoder, llm_listwise

CANDIDATES = 30  # first stage: what the reranker reads
ANSWER_MODELS = {
    CHAT_MODEL: "Gemini Flash",
    LABELING_MODEL: "Gemini Argon",
    "google_genai:gemma-4-31b-it": "Gemma 4 31B",
}
MODES = {
    "ibrida": RetrievalMode.HYBRID,
    "per significato": RetrievalMode.DENSE,
    "per parole": RetrievalMode.SPARSE,
}
EVIDENCE = {"text": "testo", "table": "tabella", "image": "immagine", "none": "assente"}
EVIDENCE_ICONS = {"text": "📄", "table": "📊", "image": "🖼️", "none": "🚫"}

st.set_page_config(page_title="Chiedi ai bandi FESR", page_icon="💬", layout="wide")


# --- Loaded once, shared by every question -------------------------------------------


@st.cache_resource(show_spinner="Carico i modelli e gli indici…")
def indexes() -> dict:
    """The baseline index (decks 09-10) and its copy with the OCR'd pages (deck 11)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embeddings, bm25 = embedding_model("bge-m3"), sparse_model("bm25")
    text = vector_store(  # on disk: built only if missing
        splitter.split_documents(load_pages()),
        embeddings,
        "bge-m3-1000-bm25",
        sparse=bm25,
    )
    ocr_pages = splitter.split_documents(
        load_pages(DATA / "processed/ocr/gemini.jsonl")
    )
    with_ocr = extended_store(
        "bge-m3-1000-bm25",
        ocr_pages,
        embeddings,
        "bge-m3-1000-bm25-ocr-gemini",
        sparse=bm25,
    )
    return {False: text, True: with_ocr}


@st.cache_resource(show_spinner="Carico il reranker…")
def reranker(name: str):
    if name == "bge-reranker (locale)":
        return cross_encoder("bge-reranker-v2-m3")
    return llm_listwise(chat_model(CHAT_MODEL, thinking_level="low", timeout=60))


@st.cache_resource
def llm(model: str, thinking: str):
    """The answering model; 'predefinito' is the setting measured in the slides."""
    if thinking == "basso" and not model.startswith("google_genai:gemma-"):
        return chat_model(model, thinking_level="low")
    return chat_model(model)


@st.cache_resource
def answer_judge():
    """A separate judge client, independent of the answering model."""
    return chat_model(JUDGE_MODEL, thinking_level="low", timeout=60)


@st.cache_data
def pages() -> dict:
    """Every page of the corpus by (url, page): where to find its picture."""
    return {
        (p["url"], p["page"]): p for p in read_jsonl(DATA / "processed/pages.jsonl")
    }


@st.cache_data
def examples() -> list[dict]:
    """The evaluation questions, with the official FAQ answer to compare."""
    eval_set = DATA / "eval"
    return read_jsonl(eval_set / "faq_questions.jsonl") + read_jsonl(
        eval_set / "image_questions.jsonl"
    )


def seconds(t: float) -> str:
    return f"{t:.1f} s".replace(".", ",")


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", ".")


# --- One question, step by step ------------------------------------------------------


def ask(question: str, settings: dict, status) -> dict:
    """Retrieve, rerank, answer: each step is written in the status box as it ends."""
    turn = {
        "question": question,
        "settings": settings,
        "steps": [],
        "chunks": [],
        "api_costs": [],
    }
    turn["faq"] = next((q for q in examples() if q["question"] == question), None)

    def step(text: str) -> None:
        turn["steps"].append(text)
        status.markdown(text)

    if settings["rag"]:
        store = search_mode(indexes()[settings["ocr"]], MODES[settings["mode"]])
        options = (
            {"filter": metadata_filter(call=settings["call"])}
            if settings["call"]
            else {}
        )
        start = time.perf_counter()
        candidates = store.as_retriever(search_kwargs={"k": CANDIDATES}).invoke(
            question, **options
        )
        where = "nel bando scelto" if settings["call"] else "in tutti i documenti"
        step(
            f"🔎 **Ricerca {settings['mode']}** {where}: {len(candidates)} chunk "
            f"candidati · {seconds(time.perf_counter() - start)}"
        )

        order = list(range(len(candidates)))
        if settings["reranker"] != "nessuno":
            score = reranker(settings["reranker"])  # loaded once, not timed
            start = time.perf_counter()
            with get_usage_metadata_callback() as rerank_usage:
                scores = score(question, candidates)
            if settings["reranker"] == "Gemini (API)":
                turn["api_costs"].append(
                    {
                        "stage": "Reranking",
                        "model": CHAT_MODEL,
                        "usd": api_cost(CHAT_MODEL, rerank_usage.usage_metadata),
                    }
                )
            order.sort(key=lambda i: scores[i], reverse=True)
            step(
                f"↕️ **Reranking** con {settings['reranker']}: riletti tutti i "
                f"{len(candidates)} · {seconds(time.perf_counter() - start)}"
            )
        turn["positions"] = order[: settings["k"]]  # where each chunk was in the search
        turn["chunks"] = [candidates[i] for i in turn["positions"]]

        prompt = PRUDENT_PROMPT if settings["prompt"] == "prudente" else PROMPT
        inputs = {"context": format_context(turn["chunks"]), "question": question}
        turn["prompt"] = prompt.invoke(inputs).to_messages()
        chain = (
            prompt | llm(settings["model"], settings["thinking"]) | StrOutputParser()
        )
    else:
        step("🚫 **Nessun documento**: il modello risponde con quello che ricorda")
        turn["prompt"] = [HumanMessage(question)]
        chain = llm(settings["model"], settings["thinking"]) | StrOutputParser()
        inputs = question

    status.update(label="Scrivo la risposta…")
    start = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        turn["answer"] = st.empty().write_stream(chain.stream(inputs))
    turn["api_costs"].append(
        {
            "stage": "Risposta",
            "model": settings["model"],
            "usd": api_cost(settings["model"], usage.usage_metadata),
        }
    )
    tokens = next(iter(usage.usage_metadata.values()), {})
    thinking = tokens.get("output_token_details", {}).get("reasoning", 0)
    step(
        f"✍️ **Risposta** di {ANSWER_MODELS[settings['model']]}: "
        f"{thousands(tokens.get('input_tokens', 0))} token letti, "
        f"{thousands(tokens.get('output_tokens', 0))} scritti"
        + (f" (di cui {thousands(thinking)} per ragionare)" if thinking else "")
        + f" · {seconds(time.perf_counter() - start)}"
    )
    if settings["judge"]:
        status.update(label="Valuto la correttezza…")
        expected = turn["faq"]["answer"] if turn["faq"] else None
        turn["judge_basis"] = (
            "Confronto con la risposta ufficiale della FAQ"
            if expected
            else (
                "Verifica rispetto ai documenti recuperati"
                if turn["chunks"]
                else "Nessun riferimento disponibile"
            )
        )
        start = time.perf_counter()
        with get_usage_metadata_callback() as judge_usage:
            try:
                reference_available = bool(expected or turn["chunks"])
                chat = answer_judge() if reference_available else None
                judgment = judge_answer(
                    chat, question, turn["answer"], turn["chunks"], expected
                )
                turn["judgment"] = judgment.model_dump()
                step(
                    f"⚖️ **Giudice {ANSWER_MODELS[JUDGE_MODEL]}**: {judgment.verdict}"
                    f" · {seconds(time.perf_counter() - start)}"
                )
            except Exception:
                turn["judge_error"] = True
                step("⚖️ **Giudizio non disponibile**: la risposta è stata conservata.")
        if reference_available:
            turn["api_costs"].append(
                {
                    "stage": "Giudice",
                    "model": JUDGE_MODEL,
                    "usd": api_cost(JUDGE_MODEL, judge_usage.usage_metadata),
                }
            )
    return turn


# --- How a turn is shown -------------------------------------------------------------


def summary(settings: dict) -> str:
    """The levers used for this answer, in one line."""
    levers = ["senza documenti"]
    if settings["rag"]:
        levers = [
            settings["mode"],
            "con OCR" if settings["ocr"] else "senza OCR",
            f"reranking: {settings['reranker']}",
            f"{settings['k']} chunk",
            f"prompt {settings['prompt']}",
        ]
    if settings["thinking"] != "predefinito":
        levers.append(f"ragionamento {settings['thinking']}")
    if settings.get("judge"):
        levers.append("con giudice")
    model = settings.get("model", CHAT_MODEL)
    return "Dietro le quinte · " + ", ".join([ANSWER_MODELS[model], *levers])


def show_sources(turn: dict) -> None:
    """Thumbnails of the cited pages, each linked to the PDF at that page."""
    by_page = {}  # several cited chunks may come from the same page
    for n in cited(turn["answer"], turn["chunks"]):
        meta = turn["chunks"][n - 1].metadata
        by_page.setdefault((meta["url"], meta["page"]), []).append(n)
    if not by_page:
        st.caption("La risposta non cita nessun chunk.")
    cited_pages = list(by_page.items())
    for row in range(0, len(cited_pages), 4):
        columns = st.columns(4)  # the last row may have fewer pages than columns
        for column, (key, refs) in zip(
            columns, cited_pages[row : row + 4], strict=False
        ):
            page, link = pages()[key], page_id(*key)  # the link opens the PDF there
            with column:
                st.image(page_picture(page, dpi=40), link=link, width="stretch")
                st.markdown(
                    f"**{' '.join(f'[{n}]' for n in sorted(refs))}** "
                    f"[{page['title']}, p. {key[1]}]({link})"
                )


def show_chunks(turn: dict) -> None:
    """What the model read: position before/after reranking, cited or not, OCR."""
    numbers = cited(turn["answer"], turn["chunks"])
    rows = [
        {
            "n": n,
            "citato": n in numbers,
            "documento": c.metadata["title"],
            "pagina": c.metadata["page"],
            "posizione nella ricerca": position + 1,
            "letto con OCR": not pages()[(c.metadata["url"], c.metadata["page"])][
                "text"
            ],
            "testo": c.page_content,
        }
        for n, (c, position) in enumerate(
            zip(turn["chunks"], turn["positions"], strict=True), start=1
        )
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={
            "testo": st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(
        "Il testo completo dei chunk, come lo legge il modello: scheda «Il prompt»."
    )


def show_prompt(turn: dict) -> None:
    """Exactly what the model received: the rules, the numbered chunks, the question."""
    for message in turn["prompt"]:
        st.caption({"system": "Istruzioni", "human": "Messaggio"}[message.type])
        st.code(message.text, language=None, wrap_lines=True, height=420)


def show_faq(turn: dict) -> None:
    faq = turn["faq"]
    st.markdown(f"**Risposta ufficiale** (FAQ, prova: {EVIDENCE[faq['evidence']]})")
    st.markdown(faq["answer"] or "_La FAQ rimanda ad altro: nei documenti non c'è._")
    if faq["gold"] and turn["chunks"]:
        found = gold_found(turn["chunks"], faq)
        st.markdown(
            ("✅ La pagina giusta è" if found else "❌ La pagina giusta non è")
            + " tra i chunk letti dal modello."
        )


def show(turn: dict) -> None:
    """A past turn, exactly as it looked when it was answered."""
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        with st.status(summary(turn["settings"]), state="complete", expanded=False):
            for text in turn["steps"]:
                st.markdown(text)
        st.markdown(link_citations(turn["answer"], turn["chunks"]))
        show_judgment(turn)
        show_cost(turn)
        details(turn)


def show_judgment(turn: dict) -> None:
    """Keep a model's assessment distinct from an independently verified fact."""
    if turn.get("judge_error"):
        st.warning("Il giudice non è riuscito a valutare questa risposta.")
    elif judgment := turn.get("judgment"):
        st.markdown(f"**⚖️ Giudizio AI: {judgment['verdict']}**")
        st.caption(f"{ANSWER_MODELS[JUDGE_MODEL]} · {turn['judge_basis']}")
        st.markdown(link_citations(judgment["reason"], turn["chunks"]))


def show_cost(turn: dict) -> None:
    """Show each API stage, without treating missing usage as zero."""
    costs = turn.get("api_costs", [])
    if not costs:
        return

    def dollars(value: float | None) -> str:
        if value is None:
            return "non disponibile"
        if value == 0:
            return "0,00 USD"
        if value < 0.000001:
            return "< 0,000001 USD"
        return f"{value:.6f} USD".replace(".", ",")

    complete = all(c["usd"] is not None for c in costs)
    total = sum(c["usd"] or 0 for c in costs) if complete else None
    st.caption(f"💰 Costo API stimato: **{dollars(total)}**")
    if len(costs) > 1:
        st.caption(" · ".join(f"{c['stage']}: {dollars(c['usd'])}" for c in costs))
    if not complete:
        st.caption("Consumi dei token non disponibili per una o più chiamate.")
    st.caption(
        f"[Listino Google standard]({PRICING_URL}), verificato il {PRICING_CHECKED}. "
        "L'eventuale piano gratuito può azzerare il costo."
    )


def details(turn: dict) -> None:
    tabs = {}
    if turn["chunks"]:
        tabs["Fonti citate"] = show_sources
        tabs[f"Chunk letti ({len(turn['chunks'])})"] = show_chunks
    tabs["Il prompt"] = show_prompt
    if turn["faq"]:
        tabs["La FAQ"] = show_faq
    for tab, show_tab in zip(st.tabs(list(tabs)), tabs.values(), strict=True):
        with tab:
            show_tab(turn)
    if turn["answer"].startswith(ABSTAIN):
        st.caption(
            "Il modello si è astenuto: nei chunk letti non ha trovato la risposta."
        )


# --- The page ------------------------------------------------------------------------

st.title("Chiedi ai bandi FESR")
indexes()  # load models and indexes now, not at the first question
st.caption(
    "La pipeline costruita nel corso. Ogni risposta mostra da dove viene. "
    "Le domande sono indipendenti: la chat non ricorda le precedenti."
)

with st.sidebar:
    if st.button("Pulisci chat", width="stretch"):
        st.session_state["turns"] = []

    calls = sorted({p["call"] for p in pages().values() if p["call"]})
    call = st.selectbox(
        "Bando",
        [None, *calls],
        format_func=lambda c: call_label(c) if c else "Tutti i documenti",
    )
    choices = [q for q in examples() if call in (None, q["call"])]
    example = st.selectbox(
        f"Domande di esempio ({len(choices)})",
        choices,
        format_func=lambda q: f"{EVIDENCE_ICONS[q['evidence']]} {q['question']}",
        help="📄 Testo · 📊 Tabella · 🖼️ Immagine · 🚫 Risposta assente dai documenti",
    )
    asked = st.button("Chiedi questa", width="stretch")

    st.header("Le variabili")
    model = st.selectbox(
        "Modello di risposta",
        list(ANSWER_MODELS),
        format_func=ANSWER_MODELS.get,
        help="Tutti via API Google. Gemma ha pesi aperti, ma qui gira via API.",
    )
    supports_thinking = not model.startswith("google_genai:gemma-")
    settings = {
        "model": model,
        "judge": st.toggle(
            "Valuta la correttezza",
            value=True,
            help="Giudice Flash: confronta con la FAQ ufficiale, se disponibile, "
            "altrimenti verifica la risposta sui documenti recuperati. Usa l'API.",
        ),
        "call": call,
        "rag": st.toggle("Usa i documenti", value=True),
        "ocr": st.toggle("Pagine-immagine lette con OCR", value=True),
        "mode": st.radio("Ricerca", list(MODES), horizontal=True),
        "reranker": st.radio(
            "Reranking", ["nessuno", "bge-reranker (locale)", "Gemini (API)"]
        ),
        "k": st.slider("Chunk al modello", 3, 20, 10),
        "prompt": st.radio(
            "Prompt",
            ["normale", "prudente"],
            horizontal=True,
            help="Prudente: risponde solo se trova la risposta completa",
        ),
        "thinking": st.radio(
            "Ragionamento del modello",
            ["predefinito", "basso"] if supports_thinking else ["predefinito"],
            horizontal=True,
            help="Basso: risposte più rapide ed economiche, non misurate nel corso",
            disabled=not supports_thinking,
        ),
    }

turns = st.session_state.setdefault("turns", [])
for turn in turns:
    show(turn)

question = st.chat_input("Fai una domanda sui bandi…")
if asked and example:
    question = example["question"]
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        turn = ask(question, settings, st.status("Cerco nei documenti…", expanded=True))
    turns.append(turn)
    st.rerun()  # redraw from the history: citations become links, sources appear
