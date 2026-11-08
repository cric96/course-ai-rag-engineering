"""Combine the page-context reports, optionally including Argon, into one chart.

uv run python scripts/plot_pipeline_comparison.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ragcourse.paths import ROOT
from ragcourse.viz import bar_chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / ".cache")
    parser.add_argument("--argon-report", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / ".cache/pipeline-comparison"
    )
    args = parser.parse_args()
    reports = [
        json.loads((args.root / filename).read_text())
        for filename in [
            "page-context.json",
            "page-context-bge-gemini.json",
            "page-context-local.json",
        ]
    ]
    reference, mixed, local = reports
    if reference["model"] != mixed["model"]:
        raise ValueError("Reference and mixed pipelines must share the generator")
    if [r["reranker"] for r in reports] != ["gemini", "bge", "bge"]:
        raise ValueError("Expected Gemini, bge and bge rerankers, respectively")
    local_name = (
        local["model"].split("/")[-1].replace("Qwen3-4B-Instruct-2507", "Qwen 4B")
    )
    precision = "NF4" if local["generation_settings"].get("load_in_4bit") else "BF16"
    variants = [
        ("Flash / Flash · chunk", reference["variants"]["chunk"]),
        ("Flash / Flash · pagine", reference["variants"]["pagine"]),
        ("bge / Flash · pagine", mixed["variants"]["pagine"]),
        (f"bge / {local_name} {precision} · pagine", local["variants"]["pagine"]),
    ]
    argon = None
    if args.argon_report:
        argon = json.loads(args.argon_report.read_text())
        if (
            argon["model"] != "google_genai:gemini-4-argon"
            or argon["reranker"] != "gemini"
        ):
            raise ValueError(
                "Expected Argon generation with the reference Gemini reranker"
            )
        for variant in ["chunk", "pagine"]:
            fields = ["question", "expected", "pages", "context_chars"]
            if {
                r["id"]: [r[field] for field in fields]
                for r in reference["variants"][variant]
            } != {
                r["id"]: [r[field] for field in fields]
                for r in argon["variants"][variant]
            }:
                raise ValueError("Argon and Flash must receive identical contexts")
        reports.append(argon)
        variants.insert(1, ("Flash / Argon · chunk", argon["variants"]["chunk"]))
        variants.insert(3, ("Flash / Argon · pagine", argon["variants"]["pagine"]))
    expected = {r["id"]: (r["question"], r["expected"]) for r in variants[0][1]}
    for report in reports:
        if (
            report["judge"] != reference["judge"]
            or report["retrieval"] != reference["retrieval"]
        ):
            raise ValueError("Reports must share judge and initial retrieval settings")
    for _, records in variants:
        if {r["id"]: (r["question"], r["expected"]) for r in records} != expected:
            raise ValueError(
                "Reports must evaluate the same questions and expected answers"
            )
    if {
        r["id"]: (r["pages"], r["context_chars"]) for r in mixed["variants"]["pagine"]
    } != {
        r["id"]: (r["pages"], r["context_chars"]) for r in local["variants"]["pagine"]
    }:
        raise ValueError("Mixed and local generators must receive identical pages")

    scores = pd.DataFrame(
        {label: {r["id"]: r["score"] for r in records} for label, records in variants}
    )
    rows = []
    for label, records in variants:
        delta = scores[label] - scores["Flash / Flash · pagine"]
        interval = np.quantile(
            np.random.default_rng(0)
            .choice(delta.to_numpy(), (10000, len(delta)))
            .mean(axis=1),
            [0.025, 0.975],
        )
        rows.append(
            {
                "pipeline": label,
                "correctness": float(scores[label].mean()),
                "abstained": sum(r["abstained"] for r in records),
                "mean_context_chars": sum(r["context_chars"] for r in records)
                / len(records),
                "delta_vs_gemini_pages": float(delta.mean()),
                "paired_bootstrap_95": interval.tolist(),
                "scores": {r["id"]: r["score"] for r in records},
            }
        )
    summary = {
        "questions": len(expected),
        "judge": reference["judge"],
        "retrieval": reference["retrieval"],
        "local_model": local["model"],
        "local_settings": local["generation_settings"],
        "local_output_limits": local["local_output_limits"],
        "local_outputs_at_limit": sum(
            r["output_tokens"] >= 2048 for r in local["variants"]["pagine"]
        ),
        "results": rows,
    }
    if argon:
        summary["generator_comparisons"] = {}
        for variant in ["chunk", "pagine"]:
            delta = (
                scores[f"Flash / Argon · {variant}"]
                - scores[f"Flash / Flash · {variant}"]
            )
            summary["generator_comparisons"][variant] = {
                "mean_delta": float(delta.mean()),
                "improved": int((delta > 0).sum()),
                "unchanged": int((delta == 0).sum()),
                "worse": int((delta < 0).sum()),
                "paired_bootstrap_95": np.quantile(
                    np.random.default_rng(0)
                    .choice(delta.to_numpy(), (10000, len(delta)))
                    .mean(axis=1),
                    [0.025, 0.975],
                ).tolist(),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    chart = bar_chart(
        scores.mean(),
        gray=[scores.columns[0]],
        fmt="{:.3f}",
        title=f"Reranker / generatore: confronto sulle stesse {len(expected)} FAQ",
    )
    if argon:
        chart.set_size_inches(12, 4.6)
    for suffix in [".svg", ".png"]:
        chart.savefig(args.output.with_suffix(suffix), bbox_inches="tight", dpi=150)
    print(pd.DataFrame(rows).drop(columns="scores").to_string(index=False))


if __name__ == "__main__":
    main()
