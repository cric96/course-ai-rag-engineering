"""Shared access to Gemini models (API key read from the project .env file)."""

from pathlib import Path

from dotenv import load_dotenv
from google import genai

# Checked with client.models.list() on 2026-11-03.
LABELING_MODEL = "gemini-4-argon"  # strongest model: builds the ground truth
GENERATION_MODEL = "gemini-3.8-flash"  # fast model: answers in the RAG pipeline
EMBEDDING_MODEL = "gemini-embedding-2"

ENV_FILE = Path(__file__).parents[2] / ".env"  # project root, also from slides/


def gemini_client() -> genai.Client:
    """Create a Gemini client using GEMINI_API_KEY from .env."""
    load_dotenv(ENV_FILE)
    return genai.Client()
