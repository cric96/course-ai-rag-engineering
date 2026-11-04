"""Pluggable models: every embedder and chat model is picked by name.

The pipeline only sees LangChain interfaces (Embeddings, BaseChatModel),
so a local model and an API are swapped by changing one string.
"""

from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, SparseEmbeddings

from ragcourse.paths import ROOT

load_dotenv(ROOT / ".env")  # API keys such as GEMINI_API_KEY, also from slides/

# "provider:model", LangChain style. Gemini IDs change fast: checked on 2026-11-03.
LABELING_MODEL = (
    "google_genai:gemini-4-argon"  # strongest model: builds the ground truth
)
CHAT_MODEL = "google_genai:gemini-3.8-flash"
JUDGE_MODEL = CHAT_MODEL  # fast and cheap; a stronger, different judge is one name away


@dataclass(frozen=True)
class LocalEmbedder:
    """A Hugging Face model and the extra encode options it expects."""

    model: str
    passage: dict = field(default_factory=dict)  # for the documents
    query: dict = field(default_factory=dict)  # for the questions


E5 = {"passage": {"prompt": "passage: "}, "query": {"prompt": "query: "}}  # e5 markers

EMBEDDERS = {
    "e5-small": LocalEmbedder("intfloat/multilingual-e5-small", **E5),  # 118M, 384 dims
    "e5-large": LocalEmbedder(
        "intfloat/multilingual-e5-large", **E5
    ),  # 560M, 1024 dims
    "bge-m3": LocalEmbedder("BAAI/bge-m3"),  # 568M, 1024 dims, long inputs
    # Qwen3 puts an instruction before each question, stored in the model as "query".
    "qwen3-0.6b": LocalEmbedder(
        "Qwen/Qwen3-Embedding-0.6B", query={"prompt_name": "query"}
    ),
}


def embedding_model(name: str = "e5-small") -> Embeddings:
    """A local embedder from EMBEDDERS, or an API one as "google_genai:<model>"."""
    if name.startswith("google_genai:"):
        return GoogleGenerativeAIEmbeddings(model=name.removeprefix("google_genai:"))
    if name not in EMBEDDERS:
        raise ValueError(f"unknown embedding model: {name}")
    spec = EMBEDDERS[name]
    return HuggingFaceEmbeddings(
        model_name=spec.model,
        encode_kwargs={**spec.passage, "normalize_embeddings": True},
        query_encode_kwargs={**spec.query, "normalize_embeddings": True},
    )


def sparse_model(name: str = "bm25") -> SparseEmbeddings:
    """Keyword vectors for hybrid search: BM25 with Italian stopwords and stemming."""
    if name == "bm25":
        return FastEmbedSparse(model_name="Qdrant/bm25", language="italian")
    raise ValueError(f"unknown sparse model: {name}")


def chat_model(name: str = CHAT_MODEL, **settings) -> BaseChatModel:
    """Any LangChain chat model, e.g. "google_genai:gemini-3.8-flash".

    Provider defaults are kept: Google advises against changing Gemini 3+ temperature.
    """
    return init_chat_model(name, **settings)
