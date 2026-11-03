"""Pluggable models: every embedder and chat model is picked by name.

The pipeline only sees LangChain interfaces (Embeddings, BaseChatModel),
so a local model and an API are swapped by changing one string.
"""

from dotenv import load_dotenv
from google import genai
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings

from ragcourse.paths import ROOT

load_dotenv(ROOT / ".env")  # API keys such as GEMINI_API_KEY, also from slides/

# Gemini IDs change fast: checked with client.models.list() on 2026-11-03.
LABELING_MODEL = "gemini-4-argon"  # strongest model: builds the ground truth
CHAT_MODEL = "google_genai:gemini-3.8-flash"  # "provider:model", LangChain style


def embedding_model(name: str = "e5-small") -> Embeddings:
    """An embedder by short name; the catalog grows as the course compares models."""
    if name == "e5-small":
        # Small multilingual model, runs locally; trained with these two markers.
        return HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small",
            encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True},
            query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True},
        )
    raise ValueError(f"unknown embedding model: {name}")


def chat_model(name: str = CHAT_MODEL) -> BaseChatModel:
    """Any LangChain chat model, e.g. "google_genai:gemini-3.8-flash"."""
    return init_chat_model(name, temperature=0)


def gemini_client() -> genai.Client:
    """Raw Gemini client, used by the scripts that build the evaluation set."""
    return genai.Client()
