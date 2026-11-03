from ragcourse.chunking import Chunk, chunk_pages, split_words


def words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_a_single_chunk():
    assert split_words("Le spese sono ammissibili.", size=200, overlap=50) == [
        "Le spese sono ammissibili."
    ]


def test_long_text_becomes_overlapping_windows():
    chunks = split_words(words(10), size=4, overlap=1)
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_last_window_may_be_shorter():
    assert split_words(words(5), size=4, overlap=0) == ["w0 w1 w2 w3", "w4"]


def test_empty_text_has_no_chunks():
    assert split_words(" \n ", size=4, overlap=1) == []


def test_chunk_pages_keeps_the_source_page_and_skips_empty_pages():
    pages = [
        {"url": "u", "page": 3, "title": "Bando", "text": "Spese ammissibili."},
        {"url": "u", "page": 4, "title": "Bando", "text": ""},
    ]
    assert chunk_pages(pages) == [Chunk("Spese ammissibili.", "u", 3, "Bando")]
