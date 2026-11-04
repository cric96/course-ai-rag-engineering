"""Project folders, valid from scripts, tests and slides alike."""

from pathlib import Path

ROOT = Path(__file__).parents[2]  # the repository root
DATA = ROOT / "data"


def project_path(path: str | Path) -> Path:
    """Paths stored in the data are relative to the project: make them work anywhere."""
    return ROOT / path  # an absolute `path` wins over ROOT
