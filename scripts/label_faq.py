"""Link each FAQ to the corpus pages that support its answer, using Gemini.

One request per call: the model reads every page of the call and all its FAQ.
Answers are cached in data/interim/labels/<call>.json, so reruns are free.

    uv run python scripts/label_faq.py              # all calls
    uv run python scripts/label_faq.py --call preciseu
"""

import argparse
import json
from pathlib import Path

from google.genai import types

from ragcourse.evalset import (
    FaqLabel,
    dedupe_faqs,
    label_pages,
    labeling_contents,
    to_questions,
)
from ragcourse.io import read_jsonl, write_jsonl
from ragcourse.models import LABELING_MODEL, gemini_client

FAQ_FILE = Path("data/raw/fesr/faq.jsonl")
PAGES_FILE = Path("data/processed/pages.jsonl")
CACHE_DIR = Path("data/interim/labels")
OUT = Path("data/eval/faq_questions.jsonl")


def label_call(client, call: str, faqs: list[dict], pages: list[dict]) -> list:
    """Ask the model which pages support each FAQ (or reuse the cached answer)."""
    cache = CACHE_DIR / f"{call}.json"
    if cache.exists():
        return [FaqLabel(**label) for label in json.loads(cache.read_text())]

    response = client.models.generate_content(
        model=LABELING_MODEL,
        contents=labeling_contents(faqs, label_pages(pages)),
        # Structured output: the SDK parses the JSON into FaqLabel objects.
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=list[FaqLabel],
        ),
    )
    labels = response.parsed
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([label.model_dump() for label in labels], indent=1))
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--call", default="", help="only calls containing this text")
    args = parser.parse_args()

    faqs = dedupe_faqs(read_jsonl(FAQ_FILE))
    pages = read_jsonl(PAGES_FILE)
    client = gemini_client()

    questions = []
    for call in sorted({faq["call"] for faq in faqs if args.call in faq["call"]}):
        call_faqs = [faq for faq in faqs if faq["call"] == call]
        call_pages = [page for page in pages if page["call"] == call]
        labels = label_call(client, call, call_faqs, call_pages)
        questions += to_questions(call_faqs, labels, label_pages(call_pages))
        print(f"{len(call_faqs):>3} FAQ labeled  {call[:70]}")

    # Stable ids, used later to join retrieval and generation results.
    for number, question in enumerate(questions, start=1):
        question["id"] = f"faq-{number:03d}"
    if not args.call:
        write_jsonl(OUT, questions)
        print(f"Saved {len(questions)} questions to {OUT}")


if __name__ == "__main__":
    main()
