"""Generate questions whose answer is only in page images (slides, infographics).

The model sees every page with (almost) no text layer and writes questions
answerable from one of them. The answer is cached: reruns are free.

    uv run python scripts/generate_image_questions.py
"""

import json
from pathlib import Path

from ragcourse.evalset import (
    ImageQuestion,
    ask_image_questions,
    label_pages,
    to_image_questions,
    visual_pages,
)
from ragcourse.io import read_jsonl, write_jsonl
from ragcourse.models import LABELING_MODEL, chat_model

PAGES_FILE = Path("data/processed/pages.jsonl")
CACHE = Path("data/interim/image_questions.json")
OUT = Path("data/eval/image_questions.jsonl")
N_QUESTIONS = 20  # a few will be discarded during manual validation


def main() -> None:
    labeled = label_pages(visual_pages(read_jsonl(PAGES_FILE)))

    if not CACHE.exists():
        model = chat_model(LABELING_MODEL)  # any LangChain chat model that reads images
        generated = ask_image_questions(model, labeled, N_QUESTIONS)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps([q.model_dump() for q in generated]))
    generated = [ImageQuestion(**q) for q in json.loads(CACHE.read_text())]

    questions = to_image_questions(generated, labeled)
    for number, question in enumerate(questions, start=1):
        question["id"] = f"img-{number:03d}"
    write_jsonl(OUT, questions)
    print(f"Saved {len(questions)} image questions to {OUT}")


if __name__ == "__main__":
    main()
