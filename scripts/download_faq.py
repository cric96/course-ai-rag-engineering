"""Download the FAQ of every FESR funding call into data/raw/fesr/faq.jsonl."""

import json
from dataclasses import asdict
from pathlib import Path

import httpx

from ragcourse.fesr import fetch_faqs

OUT = Path("data/raw/fesr/faq.jsonl")


def main() -> None:
    with httpx.Client(timeout=30) as client:
        faqs = fetch_faqs(client)

    # One JSON object per line: easy to read back and to diff.
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as file:
        for faq in faqs:
            file.write(json.dumps(asdict(faq), ensure_ascii=False) + "\n")

    calls = {faq.call for faq in faqs}
    print(f"Saved {len(faqs)} FAQ from {len(calls)} calls to {OUT}")


if __name__ == "__main__":
    main()
