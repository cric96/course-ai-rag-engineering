from pathlib import Path

from ragcourse.paths import DATA, ROOT, project_path


def test_root_is_the_project_folder():
    assert (ROOT / "pyproject.toml").exists()
    assert DATA == ROOT / "data"


def test_project_paths_resolve_from_any_folder_absolute_ones_stay():
    assert (
        project_path("data/raw/x.pdf") == ROOT / "data/raw/x.pdf"
    )  # slides run in slides/
    assert project_path("/tmp/x.pdf") == Path("/tmp/x.pdf")
