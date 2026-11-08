from ragcourse.evalset import (
    FaqPageMatch,
    pagewise_labeling_message,
    summarize_pagewise,
)


def test_pagewise_input_contains_one_page_and_all_faq_answers():
    faqs = [
        {"question": "Chi?", "answer": "Le imprese"},
        {"question": "Quando?", "answer": "Entro giugno"},
    ]
    page = {"title": "Bando", "page": 7, "text": "Una pagina di testo. " * 5}

    message = pagewise_labeling_message(faqs, "P12", page)

    assert len(message.content) == 3
    assert "[P12] Bando, p. 7" in message.content[1]["text"]
    assert "[1] Q: Chi?\nA: Le imprese" in message.content[-1]["text"]
    assert "[2] Q: Quando?\nA: Entro giugno" in message.content[-1]["text"]


def test_summary_compares_associations_and_keeps_partial_evidence_separate():
    baseline = [{"faq": 1, "pages": ["P1"]}, {"faq": 2, "pages": []}]
    records = [
        {"label": "P1", "matches": [match(1, "direct")]},
        {"label": "P2", "matches": [match(2, "partial")]},
    ]

    summary = summarize_pagewise(2, baseline, records)

    assert summary["baseline_associations"] == 1
    assert summary["pagewise_associations"] == 2
    assert summary["shared_associations"] == 1
    assert summary["added_associations"] == 1
    assert summary["removed_associations"] == 0
    assert summary["baseline_covered_faqs"] == 1
    assert summary["pagewise_covered_faqs"] == 2
    assert summary["pagewise_direct_faqs"] == 1
    assert summary["newly_covered_faqs"] == [2]


def test_summary_deduplicates_links_and_reports_baseline_pages_not_visited():
    baseline = [{"faq": 1, "pages": ["P1", "P9"]}]
    records = [{"label": "P1", "matches": [match(1, "direct"), match(1, "direct")]}]

    summary = summarize_pagewise(1, baseline, records)

    assert summary["pagewise_associations"] == 1
    assert summary["baseline_unvisited_associations"] == 1
    assert summary["removed_associations"] == 0


def match(faq, support):
    return FaqPageMatch(
        faq=faq,
        support=support,
        evidence="text",
        quote="Una frase.",
        reason="Una prova.",
    ).model_dump()
