"""Generate questions whose answer is only in page images (slides, infographics).

The model sees every page with (almost) no text layer and writes questions
answerable from one of them. The answer is cached: reruns are free.

    uv run python scripts/generate_image_questions.py [--count-tokens]
"""

import argparse
import json
from pathlib import Path

from google.genai import types

from ragcourse.evalset import (
    ImageQuestion,
    image_question_contents,
    label_pages,
    to_image_questions,
    visual_pages,
)
from ragcourse.io import read_jsonl, write_jsonl
from ragcourse.models import LABELING_MODEL, gemini_client

PAGES_FILE = Path("data/processed/pages.jsonl")
CACHE = Path("data/interim/image_questions.json")
OUT = Path("data/eval/image_questions.jsonl")
N_QUESTIONS = 20  # a few will be discarded during manual validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-tokens", action="store_true")
    args = parser.parse_args()

    labeled = label_pages(visual_pages(read_jsonl(PAGES_FILE)))
    contents = image_question_contents(labeled, N_QUESTIONS)
    client = gemini_client()

    if args.count_tokens:
        count = client.models.count_tokens(model=LABELING_MODEL, contents=contents)
        print(f"{len(labeled)} visual pages, {count.total_tokens:,} tokens")
        return

    if not CACHE.exists():
        response = client.models.generate_content(
            model=LABELING_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=list[ImageQuestion],
            ),
        )
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps([q.model_dump() for q in response.parsed]))
    generated = [ImageQuestion(**q) for q in json.loads(CACHE.read_text())]

    questions = to_image_questions(generated, labeled)
    for number, question in enumerate(questions, start=1):
        question["id"] = f"img-{number:03d}"
    write_jsonl(OUT, questions)
    print(f"Saved {len(questions)} image questions to {OUT}")


if __name__ == "__main__":
    main()
