from pathlib import Path

from ragcourse.io import read_jsonl, write_jsonl


def test_jsonl_round_trip_keeps_accents(tmp_path: Path):
    rows = [{"question": "Qual è la scadenza?"}, {"question": "Chi può partecipare?"}]
    path = tmp_path / "nested" / "faq.jsonl"

    write_jsonl(path, rows)

    assert read_jsonl(path) == rows
    assert "è" in path.read_text(encoding="utf-8")  # not escaped as è
