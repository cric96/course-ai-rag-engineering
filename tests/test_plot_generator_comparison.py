import copy
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "plot_generator_comparison",
    Path(__file__).parents[1] / "scripts/plot_generator_comparison.py",
)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


def reports():
    def record(name, abstained, score):
        return {
            "id": name,
            "answer": "Answer",
            "abstained": abstained,
            "scores": {
                "correttezza": score,
                "fedeltà": score,
                "pertinenza": score,
                "richiamo del contesto": 0.1,
            },
        }

    reference = {
        "sample_seed": 0,
        "judge": "flash",
        "retrieval": "fixed",
        "prompt": "fixed",
        "questions": [{"id": name, "context": "Same source"} for name in ["a", "b"]],
        "runs": {
            "baseline": {
                "model": "flash",
                "records": [record("a", False, 0.2), record("b", True, 0.4)],
            },
            "candidate": {
                "model": "argon",
                "records": [record("a", False, 0.5), record("b", False, 0.9)],
            },
        },
        "comparison": {"correctness_delta": 0.4},
    }
    other = copy.deepcopy(reference)
    other["runs"]["candidate"] = {
        "model": "open",
        "records": [record("a", False, 0.3), record("b", False, 1)],
    }
    return reference, other


def test_three_models_use_common_answers_only_for_fidelity_and_relevancy():
    summary, table = plot.combine_reports(*reports())

    assert summary["common_answered"] == {"count": 1, "ids": ["a"]}
    assert table.loc["correttezza", "flash"] == pytest.approx(0.3)
    assert table.loc["correttezza", "open"] == pytest.approx(0.65)
    assert table.loc["fedeltà", "argon"] == 0.5
    assert table.loc["pertinenza", "open"] == 0.3
    assert (table.loc["richiamo del contesto"] == 0.1).all()


@pytest.mark.parametrize("change", ["context", "judge", "baseline"])
def test_incompatible_reports_cannot_be_combined(change):
    reference, other = reports()
    if change == "context":
        other["questions"][0]["context"] = "A different source"
    elif change == "judge":
        other["judge"] = "another judge"
    else:
        other["runs"]["baseline"]["records"][0]["scores"]["correttezza"] = 1

    with pytest.raises(ValueError):
        plot.combine_reports(reference, other)
