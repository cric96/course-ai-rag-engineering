"""Compare cached whole-call FAQ annotations with one request per page.

uv run python scripts/compare_pagewise_labels.py --limit 6
uv run python scripts/compare_pagewise_labels.py
"""

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ragcourse.costs import api_cost
from ragcourse.evalset import (
    FaqPageMatches,
    dedupe_faqs,
    label_pages,
    pagewise_labeling_message,
    summarize_pagewise,
)
from ragcourse.io import read_jsonl
from ragcourse.models import LABELING_MODEL, chat_model
from ragcourse.paths import DATA, ROOT


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def quote_found(quote: str, text: str) -> bool:
    """Check literal evidence after whitespace normalization, not its relevance."""
    return bool(quote.strip()) and " ".join(quote.split()) in " ".join(text.split())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--call", default="sostegno-allo-sviluppo-delle-startup-innovative"
    )
    parser.add_argument("--model", default=LABELING_MODEL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".cache/pagewise-label-comparison.json",
    )
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers and limit must be positive")
    faqs = [
        q
        for q in dedupe_faqs(read_jsonl(DATA / "raw/fesr/faq.jsonl"))
        if q["call"] == args.call
    ]
    pages = label_pages(
        [
            p
            for p in read_jsonl(DATA / "processed/pages.jsonl")
            if p["call"] == args.call
        ]
    )
    if not faqs or not pages:
        parser.error("No FAQ or pages for this exact call slug")
    baseline = json.loads((DATA / "interim/labels" / f"{args.call}.json").read_text())
    original_questions = {
        q["question"]: q
        for q in read_jsonl(DATA / "eval/faq_questions.jsonl")
        if q["call"] == args.call
    }
    # Page labels must still map to the same source pages as the original run.
    for entry in baseline:
        question = original_questions[faqs[entry["faq"] - 1]["question"]]
        old = {(p["url"], p["page"]) for p in question["gold"]}
        current = {(pages[p]["url"], pages[p]["page"]) for p in entry["pages"]}
        if old != current:
            raise ValueError("Baseline page order changed; cannot compare annotations")

    chat = chat_model(args.model, timeout=120, max_retries=2)
    structured = chat.with_structured_output(FaqPageMatches, include_raw=True)
    selected = list(pages.items())[: args.limit]
    records, errors = [], []
    cache_hits, new_cost = 0, 0.0
    started = time.perf_counter()
    report = {
        "call": args.call,
        "model": args.model,
        "baseline_model": LABELING_MODEL,
        "total_pages": len(pages),
        "requested_pages": len(selected),
        "baseline_page_limit_per_faq": 5,
        "pagewise_page_limit_per_faq": None,
        "interpretation": "Annotation agreement, not accuracy against human labels.",
        "cost_note": "estimated_usd includes original costs of cached pages.",
        "faqs": [dict(q, id=original_questions[q["question"]]["id"]) for q in faqs],
        "baseline": baseline,
        "records": records,
        "errors": errors,
    }

    def faq_comparison():
        def order(labels):
            return sorted(labels, key=lambda label: int(label[1:]))

        rows = []
        for n, faq in enumerate(report["faqs"], 1):
            old = {p for entry in baseline if entry["faq"] == n for p in entry["pages"]}
            direct, partial = set(), set()
            for record in records:
                for match in record["matches"]:
                    if match["faq"] == n:
                        target = direct if match["support"] == "direct" else partial
                        target.add(record["label"])
            rows.append(
                {
                    "id": faq["id"],
                    "faq": n,
                    "question": faq["question"],
                    "baseline_pages": order(old),
                    "direct_pages": order(direct),
                    "partial_pages": order(partial),
                    "added_pages": order((direct | partial) - old),
                    "removed_pages": order(old - (direct | partial)),
                }
            )
        return rows

    def annotate(item):
        label, page = item
        message = pagewise_labeling_message(faqs, label, page)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "model": args.model,
                    "input": message.model_dump(),
                    "schema": FaqPageMatches.model_json_schema(),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        cache = DATA / "interim/labels-pagewise" / args.call / f"{fingerprint}.json"
        if cache.exists():
            return json.loads(cache.read_text()), True
        begin = time.perf_counter()
        response = structured.invoke([message])
        parsed = response["parsed"]
        if parsed is None:
            raise ValueError("The response did not match the output schema")
        if any(not 1 <= match.faq <= len(faqs) for match in parsed.matches):
            raise ValueError("The model returned a nonexistent FAQ number")
        usage = response["raw"].usage_metadata
        record = {
            "label": label,
            "url": page["url"],
            "page": page["page"],
            "title": page["title"],
            "input_sha256": fingerprint,
            "image": len(page["text"]) < 50,
            "matches": [
                dict(
                    m.model_dump(),
                    quote_found=(
                        quote_found(m.quote, page["text"])
                        if len(page["text"]) >= 50
                        else None
                    ),
                )
                for m in parsed.matches
            ],
            "usage": usage,
            "usd": api_cost(args.model, {args.model: usage} if usage else {}),
            "seconds": time.perf_counter() - begin,
        }
        save(cache, record)
        return record, False

    def checkpoint():
        records.sort(key=lambda r: int(r["label"][1:]))
        report["summary"] = summarize_pagewise(len(faqs), baseline, records)
        report["faq_comparison"] = faq_comparison()
        report["complete"] = len(records) == len(pages)
        report["wall_seconds"] = time.perf_counter() - started
        report["estimated_usd"] = sum(r["usd"] or 0 for r in records)
        report["cache_hits"] = cache_hits
        report["new_api_calls"] = len(records) - cache_hits
        report["new_calls_estimated_usd"] = new_cost
        report["cost_complete"] = all(r["usd"] is not None for r in records)
        report["usage"] = {
            key: sum((r["usage"] or {}).get(key, 0) for r in records)
            for key in ["input_tokens", "output_tokens"]
        }
        save(args.output, report)

    print(
        f"{len(faqs)} FAQ, {len(selected)}/{len(pages)} pages, {args.model}", flush=True
    )
    checkpoint()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(annotate, item): item[0] for item in selected}
        for done, future in enumerate(as_completed(jobs), 1):
            label = jobs[future]
            try:
                record, cached = future.result()
                records.append(record)
                if cached:
                    cache_hits += 1
                else:
                    new_cost += record["usd"] or 0
            except Exception as error:
                errors.append({"label": label, "error": type(error).__name__})
                print(f"{label}: {type(error).__name__}", flush=True)
            checkpoint()
            if done % 5 == 0 or done == len(selected):
                print(
                    f"{done}/{len(selected)} pages; {len(errors)} errors; "
                    f"USD {report['estimated_usd']:.4f}",
                    flush=True,
                )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    if errors:
        raise SystemExit("Incomplete run: failed pages can be retried from the cache")


if __name__ == "__main__":
    main()
