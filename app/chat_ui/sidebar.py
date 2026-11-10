"""The sidebar: call filter, example questions and the pipeline levers."""

import streamlit as st

from chat_ui.pipeline import Settings
from chat_ui.resources import (
    ANSWER_MODELS,
    MODES,
    RERANKERS,
    examples,
    pages,
    supports_thinking,
)
from ragcourse.chat import call_label

EVIDENCE_ICONS = {"text": "📄", "table": "📊", "image": "🖼️", "none": "🚫"}


def sidebar() -> tuple[Settings, str | None]:
    """The chosen settings, and the example question if 'Chiedi questa' was clicked."""
    with st.sidebar:
        if st.button("Pulisci chat", width="stretch"):
            st.session_state["turns"] = []
        call = choose_call()
        example = choose_example(call)
        st.header("Le variabili")
        settings = choose_levers(call)
    return settings, example


def choose_call() -> str | None:
    calls = sorted({p["call"] for p in pages().values() if p["call"]})
    return st.selectbox(
        "Bando",
        [None, *calls],
        format_func=lambda c: call_label(c) if c else "Tutti i documenti",
    )


def choose_example(call: str | None) -> str | None:
    choices = [q for q in examples() if call in (None, q["call"])]
    example = st.selectbox(
        f"Domande di esempio ({len(choices)})",
        choices,
        format_func=lambda q: f"{EVIDENCE_ICONS[q['evidence']]} {q['question']}",
        help="📄 Testo · 📊 Tabella · 🖼️ Immagine · 🚫 Risposta assente dai documenti",
    )
    asked = st.button("Chiedi questa", width="stretch")
    return example["question"] if asked and example else None


def choose_levers(call: str | None) -> Settings:
    model = st.selectbox(
        "Modello di risposta",
        list(ANSWER_MODELS),
        format_func=ANSWER_MODELS.get,
        help="Tutti via API Google. Gemma ha pesi aperti, ma qui gira via API.",
    )
    thinking = supports_thinking(model)
    return Settings(
        model=model,
        judge=st.toggle(
            "Valuta la correttezza",
            value=True,
            help="Giudice Flash: confronta con la FAQ ufficiale, se disponibile, "
            "altrimenti verifica la risposta sui documenti recuperati. Usa l'API.",
        ),
        call=call,
        rag=st.toggle("Usa i documenti", value=True),
        ocr=st.toggle("Pagine-immagine lette con OCR", value=True),
        mode=st.radio("Ricerca", list(MODES), horizontal=True),
        reranker=st.radio("Reranking", RERANKERS),
        k=st.slider("Chunk al modello", 3, 20, 10),
        prompt=st.radio(
            "Prompt",
            ["normale", "prudente"],
            horizontal=True,
            help="Prudente: risponde solo se trova la risposta completa",
        ),
        thinking=st.radio(
            "Ragionamento del modello",
            ["predefinito", "basso"] if thinking else ["predefinito"],
            horizontal=True,
            help="Basso: risposte più rapide ed economiche, non misurate nel corso",
            disabled=not thinking,
        ),
    )
