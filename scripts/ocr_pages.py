"""Read the pages that are only pictures (no text layer) with one OCR engine.

One page per line in data/processed/ocr/<engine>.jsonl; a rerun continues where
it stopped, so a long local run can be interrupted.

    uv run python scripts/ocr_pages.py tesseract   # classic OCR, local CPU
    uv run python scripts/ocr_pages.py qwen3-vl    # small vision model, local GPU
    uv run python scripts/ocr_pages.py gemini      # Gemini Flash, via API
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from ragcourse.cache import cache_llm_calls
from ragcourse.evalset import visual_pages
from ragcourse.io import read_jsonl
from ragcourse.models import chat_model
from ragcourse.ocr import local_vlm_reader, tesseract, vlm_reader
from ragcourse.paths import DATA

PAGES = DATA / "processed/pages.jsonl"
OCR_DIR = DATA / "processed/ocr"
ENGINES = {  # how to build the reader, and how many pages it reads at once
    "tesseract": (lambda: tesseract, 4),
    "qwen3-vl": (local_vlm_reader, 1),
    "gemini": (lambda: vlm_reader(chat_model()), 8),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=ENGINES)
    engine = parser.parse_args().engine

    out = OCR_DIR / f"{engine}.jsonl"
    done = {(r["url"], r["page"]) for r in read_jsonl(out)} if out.exists() else set()
    todo = [
        p for p in visual_pages(read_jsonl(PAGES)) if (p["url"], p["page"]) not in done
    ]
    print(f"{engine}: {len(done)} pages already read, {len(todo)} to go", flush=True)

    cache_llm_calls()  # Gemini answers are kept: a rerun costs nothing
    make, parallel = ENGINES[engine]
    read = make()

    def timed(page: dict) -> dict:
        start = time.perf_counter()
        text = read(page)
        return {**page, "text": text, "seconds": round(time.perf_counter() - start, 2)}

    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(parallel) as pool, out.open("a", encoding="utf-8") as file:
        for n, record in enumerate(pool.map(timed, todo), start=1):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            file.flush()  # every page is saved as soon as it is read
            if n % 25 == 0 or n == len(todo):
                print(f"{engine}: {n}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
