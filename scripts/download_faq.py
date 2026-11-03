"""Download the FAQ of every FESR funding call into data/raw/fesr/faq.jsonl."""

from dataclasses import asdict
from pathlib import Path

import httpx

from ragcourse.fesr import fetch_faqs
from ragcourse.io import write_jsonl

OUT = Path("data/raw/fesr/faq.jsonl")


def main() -> None:
    with httpx.Client(timeout=30) as client:
        faqs = fetch_faqs(client)

    write_jsonl(OUT, [asdict(faq) for faq in faqs])

    calls = {faq.call for faq in faqs}
    print(f"Saved {len(faqs)} FAQ from {len(calls)} calls to {OUT}")


if __name__ == "__main__":
    main()
