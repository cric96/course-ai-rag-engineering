"""Build the evaluation set: FAQ questions linked to the pages that answer them."""

import mimetypes
from pathlib import Path
from typing import Literal

from google.genai import types
from pydantic import BaseModel

from ragcourse.parsing import render_page

MIN_TEXT_CHARS = 50  # below this, a page is shown to the model as an image

LABELING_PROMPT = """\
You are building the ground truth to evaluate a search system over Italian documents.
Below are ALL the pages of the documents of one funding call. Each page starts with a
label like [P12]; pages with little text are given as images.
After the pages there is a numbered list of FAQ: a question and its official answer.

For every FAQ return:
- pages: labels of the pages whose content supports the official answer, i.e. the pages
  someone must read to give that answer. Only the necessary ones, at most 5.
  Return an empty list if no page supports the answer.
- evidence: where the supporting content is: "text" (running text), "table",
  "image" (figure, screenshot, infographic, scanned page) or "none".
"""


class FaqLabel(BaseModel):
    """What the labeling model returns for one FAQ."""

    faq: int  # FAQ number in the prompt
    pages: list[str]  # supporting page labels, e.g. ["P12", "P13"]
    evidence: Literal["text", "table", "image", "none"]


def dedupe_faqs(faqs: list[dict]) -> list[dict]:
    """Drop questions repeated within the same call, keeping the first."""
    seen, unique = set(), []
    for faq in faqs:
        key = (faq["call"], faq["question"].strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(faq)
    return unique


def label_pages(pages: list[dict]) -> dict[str, dict]:
    """Give each page a short label (P1, P2, ...) that the model can cite."""
    return {f"P{number}": page for number, page in enumerate(pages, start=1)}


def page_parts(label: str, page: dict) -> list:
    """Prompt parts for one page: its text, or its image when text is missing."""
    header = f"[{label}] {page['title']}, p. {page['page']}"
    if len(page["text"]) >= MIN_TEXT_CHARS:
        return [f"{header}\n{page['text']}"]
    return [f"{header} (image)", page_image(page)]


def page_image(page: dict) -> types.Part:
    """The page as an image: image files as they are, PDF pages rendered."""
    path = Path(page["path"])
    if page["kind"] == "image":
        mime_type = mimetypes.guess_type(path)[0] or "image/png"
        return types.Part.from_bytes(data=path.read_bytes(), mime_type=mime_type)
    png = render_page(path, page["page"])
    return types.Part.from_bytes(data=png, mime_type="image/png")


def labeling_contents(faqs: list[dict], labeled_pages: dict[str, dict]) -> list:
    """The whole request: instructions, every page, then the numbered FAQ."""
    contents = [LABELING_PROMPT]
    for label, page in labeled_pages.items():
        contents += page_parts(label, page)
    numbered = [
        f"[{n}] Q: {faq['question']}\nA: {faq['answer']}"
        for n, faq in enumerate(faqs, start=1)
    ]
    contents.append("FAQ:\n\n" + "\n\n".join(numbered))
    return contents


def to_questions(
    faqs: list[dict], labels: list[FaqLabel], labeled_pages: dict[str, dict]
) -> list[dict]:
    """Attach gold pages to each FAQ; labels the model invented are ignored."""
    by_number = {label.faq: label for label in labels}
    questions = []
    for n, faq in enumerate(faqs, start=1):
        label = by_number.get(n, FaqLabel(faq=n, pages=[], evidence="none"))
        gold = [
            {"url": labeled_pages[p]["url"], "page": labeled_pages[p]["page"]}
            for p in label.pages
            if p in labeled_pages
        ]
        questions.append(
            {**faq, "source": "faq", "evidence": label.evidence, "gold": gold}
        )
    return questions
