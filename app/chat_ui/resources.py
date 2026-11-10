"""Models, indexes and data loaded once and shared by every question."""

import streamlit as st
from langchain_qdrant import RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragcourse.corpus import load_pages
from ragcourse.index import extended_store, vector_store
from ragcourse.io import read_jsonl
from ragcourse.models import (
    CHAT_MODEL,
    JUDGE_MODEL,
    LABELING_MODEL,
    chat_model,
    embedding_model,
    sparse_model,
)
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
NO_RERANKER, LOCAL_RERANKER, GEMINI_RERANKER = RERANKERS = [
    "nessuno",
    "bge-reranker (locale)",
    "Gemini (API)",
]


def supports_thinking(model: str) -> bool:
    return not model.startswith("google_genai:gemma-")


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
    if name == LOCAL_RERANKER:
        return cross_encoder("bge-reranker-v2-m3")
    return llm_listwise(chat_model(CHAT_MODEL, thinking_level="low", timeout=60))


@st.cache_resource
def llm(model: str, thinking: str):
    """The answering model; 'predefinito' is the setting measured in the slides."""
    if thinking == "basso" and supports_thinking(model):
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


def faq_for(question: str) -> dict | None:
    """The FAQ entry asking exactly this question, if any."""
    return next((q for q in examples() if q["question"] == question), None)
