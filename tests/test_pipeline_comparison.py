import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pipeline_comparison",
    Path(__file__).parents[1] / "scripts/plot_pipeline_comparison.py",
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_argon_comparison_pairs_the_same_context_and_keeps_the_flash_reference(
    tmp_path, monkeypatch
):
    records = [
        {
            "id": str(i),
            "question": f"Question {i}",
            "expected": "Answer",
            "pages": ["p1"],
            "context_chars": 100,
            "score": 0.4,
            "abstained": False,
            "output_tokens": 20,
        }
        for i in range(2)
    ]
    reference = {
        "model": "flash",
        "judge": "flash",
        "retrieval": "fixed",
        "reranker": "gemini",
        "variants": {"chunk": records, "pagine": records},
    }
    mixed = {**reference, "reranker": "bge"}
    local = {
        **mixed,
        "model": "local:qwen",
        "generation_settings": {"load_in_4bit": True},
        "local_output_limits": [1024, 2048],
    }
    argon = copy.deepcopy(reference)
    argon["model"] = "google_genai:gemini-4-argon"
    for variant in argon["variants"].values():
        for record in variant:
            record["score"] = 0.6
    for name, data in [
        ("page-context.json", reference),
        ("page-context-bge-gemini.json", mixed),
        ("page-context-local.json", local),
        ("argon.json", argon),
    ]:
        (tmp_path / name).write_text(json.dumps(data))
    output = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_pipeline_comparison.py",
            "--root",
            str(tmp_path),
            "--argon-report",
            str(tmp_path / "argon.json"),
            "--output",
            str(output),
        ],
    )

    experiment.main()
    summary = json.loads(output.with_suffix(".json").read_text())
    assert len(summary["results"]) == 6
    rows = {row["pipeline"]: row for row in summary["results"]}
    assert rows["Flash / Flash · pagine"]["delta_vs_gemini_pages"] == 0
    assert rows["Flash / Argon · pagine"]["delta_vs_gemini_pages"] == pytest.approx(0.2)
    assert summary["generator_comparisons"]["chunk"]["mean_delta"] == pytest.approx(0.2)

    argon["variants"]["pagine"][0]["context_chars"] += 1
    (tmp_path / "argon.json").write_text(json.dumps(argon))
    with pytest.raises(ValueError, match="identical contexts"):
        experiment.main()
