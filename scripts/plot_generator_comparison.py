"""Combine the Argon and open-model experiments with their common Flash baseline."""

import argparse
import json
from pathlib import Path

import pandas as pd

from ragcourse.paths import ROOT
from ragcourse.viz import compare_chart

LABELS = {
    "google_genai:gemini-3.8-flash": "Gemini 3.8 Flash",
    "google_genai:gemini-4-argon": "Gemini 4 Argon",
    "google_genai:gemma-4-31b-it": "Gemma 4 31B · open",
}


def combine_reports(reference: dict, other: dict) -> tuple[dict, pd.DataFrame]:
    """Require identical inputs and baseline results before combining three models."""
    for field in ["sample_seed", "judge", "retrieval", "prompt"]:
        if reference[field] != other[field]:
            raise ValueError(f"Reports must share {field}")
    first = {q["id"]: q for q in reference["questions"]}
    second = {q["id"]: q for q in other["questions"]}
    if first != second or len(first) != len(reference["questions"]):
        raise ValueError("Reports must share questions, expected answers and contexts")
    if len(second) != len(other["questions"]):
        raise ValueError("Duplicate questions in the open-model report")
    baseline = reference["runs"]["baseline"]
    if baseline["model"] != other["runs"]["baseline"]["model"]:
        raise ValueError("Reports must share the baseline model")

    def baseline_values(run):
        return {
            r["id"]: (r["answer"], r["abstained"], r["scores"]) for r in run["records"]
        }

    if baseline_values(baseline) != baseline_values(other["runs"]["baseline"]):
        raise ValueError("Reports must share baseline answers and scores")
    runs = [baseline, reference["runs"]["candidate"], other["runs"]["candidate"]]
    if len({run["model"] for run in runs}) != 3:
        raise ValueError("Expected three distinct generators")
    indexed = {}
    for run in runs:
        records = {r["id"]: r for r in run["records"]}
        if set(records) != set(first) or len(records) != len(run["records"]):
            raise ValueError("Every generator must answer the same questions once")
        indexed[run["model"]] = records
    common = sorted(
        set.intersection(
            *(
                {key for key, record in records.items() if not record["abstained"]}
                for records in indexed.values()
            )
        )
    )
    if not common:
        raise ValueError("No questions answered by all three generators")
    summary = {
        "questions": len(first),
        "question_ids": list(first),
        "judge": reference["judge"],
        "retrieval": reference["retrieval"],
        "sample_seed": reference["sample_seed"],
        "common_answered": {"count": len(common), "ids": common},
        "runs": {},
        "comparison_vs_flash": {
            reference["runs"]["candidate"]["model"]: reference["comparison"],
            other["runs"]["candidate"]["model"]: other["comparison"],
        },
    }
    for model, records in indexed.items():
        means = {}
        for metric in ["correttezza", "fedeltà", "pertinenza", "richiamo del contesto"]:
            ids = common if metric in ["fedeltà", "pertinenza"] else list(first)
            means[metric] = sum(records[key]["scores"][metric] for key in ids) / len(
                ids
            )
        summary["runs"][model] = {
            "means": means,
            "answers_given": sum(not r["abstained"] for r in records.values()),
        }
    summary["context_recall_zero"] = sum(
        record["scores"]["richiamo del contesto"] == 0
        for record in indexed[baseline["model"]].values()
    )
    table = pd.DataFrame(
        {
            LABELS.get(model, model): values["means"]
            for model, values in summary["runs"].items()
        }
    )
    return summary, table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--argon-report",
        type=Path,
        default=ROOT / ".cache/generator-comparison-argon.json",
    )
    parser.add_argument(
        "--open-report",
        type=Path,
        default=ROOT / ".cache/generator-comparison-gemma.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / ".cache/generator-comparison-open",
    )
    args = parser.parse_args()
    summary, table = combine_reports(
        json.loads(args.argon_report.read_text()),
        json.loads(args.open_report.read_text()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    )
    chart = compare_chart(
        table,
        fmt="{:.2f}",
        title=f"Stessi 5 chunk e prompt · {summary['questions']} FAQ",
    )
    chart.set_size_inches(12.5, 4.8)
    for suffix in [".svg", ".png"]:
        chart.savefig(args.output.with_suffix(suffix), bbox_inches="tight", dpi=150)
    print(table.to_string())
    print(f"Risposte comuni: {summary['common_answered']['count']}")


if __name__ == "__main__":
    main()
