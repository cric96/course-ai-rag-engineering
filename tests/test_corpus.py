from pathlib import Path

from ragcourse.corpus import load_pages

from ragcourse.io import write_jsonl


def page(text: str, number: int) -> dict:
    return {
        "url": "u",
        "page": number,
        "title": "Bando",
        "call": "bando-a",
        "text": text,
        "path": "x.pdf",
        "kind": "file",
    }


def test_load_pages_makes_a_document_for_each_page_with_text(tmp_path: Path):
    path = tmp_path / "pages.jsonl"
    write_jsonl(path, [page("Spese ammissibili.", 3), page("", 4)])

    [doc] = load_pages(path)

    assert doc.page_content == "Spese ammissibili."
    assert doc.metadata == {"url": "u", "page": 3, "title": "Bando", "call": "bando-a"}
