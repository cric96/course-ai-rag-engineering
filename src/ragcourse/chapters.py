"""One presentation, many chapters: each chapter starts from a clean notebook."""

import gc
import importlib.util

from qdrant_client import QdrantClient

from ragcourse.cache import cache_searches


def close_indexes() -> int:
    """Close every open Qdrant client: a local index folder opens in one client only."""
    # type(), not isinstance(): a dead weakref proxy raises ReferenceError when touched
    clients = [obj for obj in gc.get_objects() if issubclass(type(obj), QdrantClient)]
    for client in clients:
        client.close()
    return len(clients)


def new_chapter(shell=None, keep: str = "percorso") -> None:
    """Close the indexes, forget every variable but `keep`, free the GPU.

    Searches are kept on disk: a re-render repeats none of them.
    """
    if shell is None:
        from IPython import get_ipython  # only inside a notebook

        shell = get_ipython()
    close_indexes()
    cache_searches()
    shell.run_line_magic("reset_selective", f"-f ^(?!{keep}$)")
    gc.collect()
    if importlib.util.find_spec("torch"):  # models group only
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
