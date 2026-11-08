from langchain_core.embeddings import DeterministicFakeEmbedding

from ragcourse.cache import cache_searches
from ragcourse.cache import searches as search_cache
from ragcourse.filters import metadata_filter
from ragcourse.visual import picture_store

PAGES = [
    {
        "url": "u",
        "page": 1,
        "title": "Slide",
        "call": "bando-a",
        "kind": "file",
        "path": "x",
    },
    {
        "url": "u",
        "page": 2,
        "title": "Slide",
        "call": "bando-b",
        "kind": "file",
        "path": "x",
    },
]


class FakePictures(DeterministicFakeEmbedding):
    """Pretends to embed page pictures: one fixed vector per page number."""

    def embed_pictures(self, pages):
        return [self.embed_query(f"page {page['page']}") for page in pages]


def test_picture_index_is_searched_by_text_like_any_langchain_store(tmp_path):
    store = picture_store(PAGES, FakePictures(size=8), "pics", root=tmp_path)

    found = store.as_retriever(search_kwargs={"k": 2}).invoke(
        "spese", filter=metadata_filter(call="bando-b")
    )

    assert [(d.metadata["url"], d.metadata["page"]) for d in found] == [("u", 2)]
    assert found[0].page_content == "[immagine] Slide, p. 2"


def test_picture_searches_are_kept_on_disk_like_text_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(search_cache, "SEARCHES", None)  # restored after the test
    cache_searches(tmp_path / "searches.sqlite")
    store = picture_store(PAGES, FakePictures(size=8), "pics", root=tmp_path)
    first = store.similarity_search("spese", k=1)

    store.client.close()  # the index is closed: only the cache can answer

    assert store.similarity_search("spese", k=1) == first
