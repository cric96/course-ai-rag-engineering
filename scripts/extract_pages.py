"""Extract the text layer of every corpus page into data/processed/pages.jsonl.

Images and image-only pages get empty text: plain extraction cannot see them.
"""

from pathlib import Path

from ragcourse.io import read_jsonl, write_jsonl
from ragcourse.parsing import extract_pages

INDEX_FILE = Path("data/raw/fesr/documents.jsonl")
OUT = Path("data/processed/pages.jsonl")


def main() -> None:
    pages = []
    for document in read_jsonl(INDEX_FILE):
        # An image is a single "page" without a text layer.
        if document["kind"] == "image":
            texts = [""]
        else:
            texts = extract_pages(Path(document["path"]))
        for number, text in enumerate(texts, start=1):
            pages.append({**document, "page": number, "text": text})

    write_jsonl(OUT, pages)
    empty = sum(not page["text"] for page in pages)
    print(f"Saved {len(pages)} pages to {OUT} ({empty} without text)")


if __name__ == "__main__":
    main()
