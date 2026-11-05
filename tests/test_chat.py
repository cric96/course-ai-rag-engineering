from langchain_core.documents import Document

from ragcourse.chat import call_label, cited, link_citations


def chunk(url: str, page: int) -> Document:
    return Document("...", metadata={"url": url, "page": page, "title": "Bando"})


CHUNKS = [chunk("https://x/bando.pdf", 4), chunk("https://x/faq.pdf", 2)]


def test_call_label_turns_the_slug_into_a_readable_name():
    slug = "sostegno-all2019innovazione-delle-imprese-anno-2024"

    assert call_label(slug) == "Sostegno all’innovazione delle imprese anno 2024"


def test_cited_lists_the_chunks_the_answer_cites_once_in_order():
    answer = "Sì [2]. Anche le microimprese [1, 2]. Vedi [2]."

    assert cited(answer, CHUNKS) == [2, 1]


def test_cited_ignores_numbers_without_a_chunk():
    assert cited("Come da [7].", CHUNKS) == []


def test_link_citations_points_each_number_to_its_pdf_page():
    linked = link_citations("Sì [2].", CHUNKS)

    assert linked == r"Sì \[[2](https://x/faq.pdf#page=2)\]."


def test_link_citations_keeps_lists_and_unknown_numbers_readable():
    linked = link_citations("Vedi [1, 7].", CHUNKS)

    assert linked == r"Vedi \[[1](https://x/bando.pdf#page=4), 7\]."
