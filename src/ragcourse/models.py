"""Pluggable models: every embedder and chat model is picked by name.

The pipeline only sees LangChain interfaces (Embeddings, BaseChatModel),
so a local model and an API are swapped by changing one string.
"""

import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, SparseEmbeddings

from ragcourse import cache
from ragcourse.cache import CachedEmbeddings, measured
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
    """A local embedder from EMBEDDERS, or an API one as "google_genai:<model>".

    Once cache_embeddings() is on, vectors are kept on disk and the model loads lazily.
    """
    if not name.startswith("google_genai:") and name not in EMBEDDERS:
        raise ValueError(f"unknown embedding model: {name}")
    if cache.EMBEDDINGS is None:
        return load_embedder(name)
    return CachedEmbeddings(lambda: load_embedder(name), name, cache.EMBEDDINGS)


def query_ms(name: str, questions: list[str]) -> float:
    """Milliseconds to embed one question with the model itself; measured once."""

    def measure() -> float:
        model = load_embedder(name)  # not the cache: its vectors take no time
        model.embed_query("riscaldamento")  # the first call also sets up the GPU
        start = time.perf_counter()
        for question in questions:
            model.embed_query(question)
        return (time.perf_counter() - start) / len(questions) * 1000

    return measured(f"ms per question: {name}", measure)


def load_embedder(name: str) -> Embeddings:
    """The embedder itself: a local model is loaded into memory (GPU if any)."""
    if name.startswith("google_genai:"):
        return GoogleGenerativeAIEmbeddings(model=name.removeprefix("google_genai:"))
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
    if name.startswith("local:"):
        return local_chat_model(name.removeprefix("local:"), **settings)
    return init_chat_model(name, **settings)


class LocalChatHuggingFace(ChatHuggingFace):
    """Include model and generation settings in LangChain's chat cache key."""

    @property
    def _identifying_params(self) -> dict:
        return {
            "model_id": self.llm.model_id,
            "model_kwargs": self.llm.model_kwargs,
            "pipeline_kwargs": self.llm.pipeline_kwargs,
        }


def local_chat_model(
    model_id: str, max_new_tokens: int = 1024, load_in_4bit: bool = False
) -> BaseChatModel:
    """A local Transformers generator, wrapped in LangChain's chat interface."""
    import torch
    from langchain_huggingface import HuggingFacePipeline

    model_settings = {"dtype": torch.bfloat16}
    placement = {"device": 0 if torch.cuda.is_available() else -1}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        model_settings["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        placement = {"device_map": "auto"}
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        **placement,
        model_kwargs=model_settings,
        pipeline_kwargs={
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "return_full_text": False,
        },
        batch_size=1,
    )
    return LocalChatHuggingFace(llm=llm, tokenizer=llm.pipeline.tokenizer)
