"""Disk caches that make re-rendering the slides fast; not part of the RAG pipeline.

Each cache is off until turned on (new_chapter() turns them on for the slides):
the pipeline code works the same with or without them.
"""

from ragcourse.cache.embeddings import cache_embeddings
from ragcourse.cache.llm import cache_llm_calls
from ragcourse.cache.measures import measured
from ragcourse.cache.scores import cache_scores, cached_scorer
from ragcourse.cache.searches import cache_searches

__all__ = [
    "cache_embeddings",
    "cache_llm_calls",
    "cache_scores",
    "cache_searches",
    "cached_scorer",
    "measured",
]
