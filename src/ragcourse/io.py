"""Read and write JSON Lines files (one JSON object per line)."""

import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write rows to `path`, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            # ensure_ascii=False keeps accented letters readable.
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read all rows from `path`."""
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]
