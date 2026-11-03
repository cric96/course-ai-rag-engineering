from ragcourse.paths import DATA, ROOT


def test_root_is_the_project_folder():
    assert (ROOT / "pyproject.toml").exists()
    assert DATA == ROOT / "data"
