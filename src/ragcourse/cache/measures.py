"""Measurements (e.g. times) taken once, then read back from disk."""

import json
from collections.abc import Callable

from ragcourse.paths import ROOT

MEASURES = ROOT / ".cache/measures.json"  # read at each call: tests point it elsewhere


def measured(name: str, measure: Callable[[], float]) -> float:
    """A measurement (e.g. a time) taken once, then read back from disk.

    Times must not be taken again on cached results: they would come out as 0.
    """
    known = json.loads(MEASURES.read_text()) if MEASURES.exists() else {}
    if name not in known:
        known[name] = measure()
        MEASURES.parent.mkdir(parents=True, exist_ok=True)
        MEASURES.write_text(json.dumps(known, indent=1, ensure_ascii=False))
    return known[name]
