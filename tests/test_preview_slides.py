import importlib.util
import json
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "preview_slides", Path(__file__).parents[1] / "scripts/preview_slides.py"
)
preview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preview)


def code(source, output=None):
    return {
        "cell_type": "code",
        "source": source,
        "outputs": [] if output is None else [{"text": output}],
        "execution_count": None if output is None else 1,
    }


def test_fast_preview_preserves_new_prose_and_saved_outputs():
    prose = {"cell_type": "markdown", "source": "The new explanation"}
    current = {"cells": [prose, code("plot(results)")]}
    cached = {"cells": [code("quarto setup"), code("plot(results)", "saved plot")]}

    result, missing = preview.reuse_outputs(current, cached)

    assert missing == 0
    assert result["cells"][0] == prose
    assert result["cells"][1]["outputs"] == [{"text": "saved plot"}]
    assert current["cells"][1]["outputs"] == []  # the source stays untouched


def test_inserted_or_changed_cells_do_not_get_old_outputs():
    current = {"cells": [code("a()"), code("new()"), code("b()"), code("changed()")]}
    cached = {
        "cells": [
            code("setup"),
            code("a()", "a"),
            code("b()", "b"),
            code("old()", "old"),
        ]
    }

    result, missing = preview.reuse_outputs(current, cached)

    assert missing == 2
    assert [c["outputs"] for c in result["cells"]] == [
        [{"text": "a"}],
        [],
        [{"text": "b"}],
        [],
    ]


def test_includes_are_expanded_relative_to_the_file(tmp_path):
    chapters = tmp_path / "chapters"
    chapters.mkdir()
    (chapters / "one.qmd").write_text("## Current prose\n{{< include two.qmd >}}")
    (chapters / "two.qmd").write_text("More prose")
    main = tmp_path / "course.qmd"
    main.write_text("# Course\n{{< include chapters/one.qmd >}}")

    assert preview.expand_includes(main) == "# Course\n## Current prose\nMore prose"


def test_fast_preview_copies_resources_into_an_independent_project(
    tmp_path, monkeypatch
):
    slides = tmp_path / "slides"
    slides.mkdir()
    main = slides / "corso.qmd"
    main.write_text("# Current prose")
    for name in preview.RESOURCES:
        (slides / name).write_text(f"resource: {name}")
    (slides / "assets").mkdir()
    (slides / "assets/plot.svg").write_text("<svg>saved chart</svg>")
    executed = slides / ".jupyter_cache/executed/run"
    executed.mkdir(parents=True)
    (executed / "base.ipynb").write_text(
        json.dumps({"cells": [code("setup"), code("plot()", "saved output")]})
    )
    fast = tmp_path / ".cache/slides-preview/corso.ipynb"
    monkeypatch.setattr(preview, "SLIDES", slides)
    monkeypatch.setattr(preview, "MAIN", main)
    monkeypatch.setattr(preview, "FAST", fast)

    def convert(command, **kwargs):
        Path(command[-1]).write_text(json.dumps({"cells": [code("plot()")]}))

    monkeypatch.setattr(preview.subprocess, "run", convert)
    preview.build_fast_preview()

    assert json.loads(fast.read_text())["cells"][0]["outputs"] == [
        {"text": "saved output"}
    ]
    for name in preview.RESOURCES:
        assert (fast.parent / name).read_bytes() == (slides / name).read_bytes()
    assert (fast.parent / "assets/plot.svg").read_text() == "<svg>saved chart</svg>"
    assert not (slides / "corso.ipynb").exists()


def test_each_fast_start_uses_a_different_project(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "FAST", tmp_path / "previews/corso.ipynb")
    monkeypatch.setattr(preview.sys, "argv", ["preview_slides.py", "--fast"])
    projects = []

    def start(target, extra, fast=False):
        assert fast
        assert target.parent.is_dir()
        projects.append(target.parent)

    monkeypatch.setattr(preview, "run_preview", start)
    preview.main()
    preview.main()

    assert projects[0] != projects[1]
    assert all(not path.exists() for path in projects)
