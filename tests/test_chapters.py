import weakref

import pytest
from qdrant_client import QdrantClient

from ragcourse.chapters import close_indexes, new_chapter


def test_close_indexes_frees_a_local_index_for_the_next_chapter(tmp_path):
    client = QdrantClient(path=str(tmp_path))

    assert close_indexes() >= 1
    QdrantClient(path=str(tmp_path)).close()  # the same folder opens again
    del client


def test_close_indexes_skips_proxies_to_deleted_objects():
    class Model:
        pass

    model = Model()
    proxy = weakref.proxy(model)  # e.g. left behind by a library
    del model  # the proxy is now dead: touching it raises ReferenceError

    close_indexes()
    del proxy


def test_new_chapter_forgets_variables_but_keeps_the_journey():
    interactive = pytest.importorskip("IPython.core.interactiveshell")
    shell = interactive.InteractiveShell.instance()
    shell.user_ns.update({"percorso": {"baseline": 0.31}, "store": object()})

    new_chapter(shell)

    assert "store" not in shell.user_ns
    assert shell.user_ns["percorso"] == {"baseline": 0.31}
