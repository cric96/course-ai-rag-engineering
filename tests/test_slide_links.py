"""Study links must point at real files and the matching source slide."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SLIDES = ROOT / "slides"


def render(source, directory):
    if not shutil.which("quarto"):
        pytest.skip("Quarto is required for the Lua filter integration checks")
    return subprocess.run(
        [
            "quarto",
            "pandoc",
            "--from=markdown",
            "--to=revealjs",
            "--slide-level=2",
            f"--lua-filter={SLIDES / 'source-links.lua'}",
        ],
        input=source,
        text=True,
        capture_output=True,
        cwd=directory,
    )


def test_links_follow_duplicate_headings_and_ignore_code_and_callouts(tmp_path):
    (tmp_path / "corso.qmd").write_text("{{< include chapter.qmd >}}\n")
    source = (
        '# Chapter\n\n## Same {data-code="src/ragcourse/models.py"}\n\n'
        "```python\n# A comment, not a slide\n```\n\n"
        "::: {.callout-note}\n## A callout title, not a slide\nNote.\n:::\n\n"
        "## Same\n\nFinal slide.\n"
    )
    (tmp_path / "chapter.qmd").write_text(source)

    result = render(source, tmp_path)

    assert result.returncode == 0, result.stderr
    assert re.findall(r"chapter.qmd#L\d+-L\d+", result.stdout) == [
        "chapter.qmd#L1-L2",
        "chapter.qmd#L3-L13",
        "chapter.qmd#L14-L16",
    ]
    assert result.stdout.count('class="slide-code-links"') == 3
    assert 'rel="noopener noreferrer"' in result.stdout
    assert 'target="_blank"' in result.stdout
    assert "/src/ragcourse/models.py" in result.stdout
    assert "data-code=" not in result.stdout

    # Inserting prose must update anchors without editing a manifest of line numbers.
    (tmp_path / "chapter.qmd").write_text("Extra prose.\n\n" + source)
    updated = render(source, tmp_path)
    assert updated.returncode == 0, updated.stderr
    assert "chapter.qmd#L5-L15" in updated.stdout
    assert "chapter.qmd#L16-L18" in updated.stdout


def test_heading_mismatch_fails_instead_of_linking_to_another_slide(tmp_path):
    (tmp_path / "corso.qmd").write_text("{{< include chapter.qmd >}}\n")
    (tmp_path / "chapter.qmd").write_text("## Expected\n")

    result = render("## Unexpected\n", tmp_path)

    assert result.returncode != 0
    assert "slide/source heading mismatch" in result.stderr


def test_repository_slide_links_cover_every_top_level_heading():
    main = (SLIDES / "corso.qmd").read_text()
    chapters = re.findall(r"\{\{< include (.*?) >\}\}", main)
    source = "\n\n".join((SLIDES / chapter).read_text() for chapter in chapters)

    # Quarto normally converts executable cell fences before invoking Pandoc.
    source = source.replace("```{python}", "```python")
    result = render(source, SLIDES)

    assert result.returncode == 0, result.stderr
    # Plain Pandoc also turns callout titles into sections; Quarto handles those
    # before the writer. Count actual slides, including chapter dividers.
    classes = re.findall(r'<section[^>]*class="([^"]*)"', result.stdout)
    actual_slides = [
        value
        for value in classes
        if "slide" in value.split() and "callout-" not in value
    ]
    assert result.stdout.count('class="slide-code-links"') == len(actual_slides)
    for chapter in chapters:
        assert f"slides/{chapter}#L1-" in result.stdout
    for paths in re.findall(r'data-code="([^"]+)"', source):
        targets = paths.split(",")
        assert len(targets) <= 3
        for target in targets:
            assert (ROOT / target).is_file(), target
            assert f"/blob/main/{target}" in result.stdout
