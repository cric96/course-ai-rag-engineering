"""Live preview of the course slides, also when a chapter changes.

quarto preview watches corso.qmd but not the chapters it includes:
when a chapter is saved, we touch corso.qmd so the preview renders again.

    uv run python scripts/preview_slides.py                # real outputs (cached)
    uv run python scripts/preview_slides.py --fast         # saved outputs, no execution
    uv run python scripts/preview_slides.py --no-execute   # text only, no outputs
"""

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from difflib import SequenceMatcher
from pathlib import Path

SLIDES = Path(__file__).resolve().parent.parent / "slides"
MAIN = SLIDES / "corso.qmd"
FAST = SLIDES.parent / ".cache/slides-preview/corso.ipynb"
RESOURCES = [
    "_quarto.yml",
    "theme.scss",
    "code.theme",
    "progress.html",
    "chapter-ids.lua",
    "source-links.lua",
]


def chapter_times() -> dict[Path, float]:
    """Last modification time of each chapter."""
    paths = [
        MAIN,
        *SLIDES.glob("_capitoli/*.qmd"),
        *(SLIDES / name for name in RESOURCES),
        *(path for path in (SLIDES / "assets").rglob("*") if path.is_file()),
    ]
    return {path: path.stat().st_mtime for path in paths}


def expand_includes(path: Path) -> str:
    """Expand chapters before converting the current slide text to a notebook."""
    return re.sub(
        r"\{\{<\s*include\s+([^\s>]+)\s*>\}\}",
        lambda match: expand_includes(path.parent / match[1]),
        path.read_text(),
    )


def reuse_outputs(notebook: dict, cached: dict) -> tuple[dict, int]:
    """Reuse outputs only for matching code cells, preserving their order."""

    def matching_source(cell: dict) -> str:
        # Showing or hiding code does not change the cell's computed output.
        return re.sub(r"(?m)^#\|\s*echo\s*:.*\n?", "", "".join(cell["source"])).strip()

    notebook = copy.deepcopy(notebook)
    current = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    previous = [cell for cell in cached["cells"] if cell["cell_type"] == "code"][1:]
    matcher = SequenceMatcher(
        a=[matching_source(cell) for cell in previous],
        b=[matching_source(cell) for cell in current],
        autojunk=False,
    )
    matched = 0
    for old, new, length in matcher.get_matching_blocks():
        for offset in range(length):
            source, target = previous[old + offset], current[new + offset]
            target["outputs"] = copy.deepcopy(source.get("outputs", []))
            target["execution_count"] = source.get("execution_count")
            matched += 1
    return notebook, len(current) - matched


def build_fast_preview(target: Path | None = None) -> None:
    """Combine today's prose with saved cell outputs, without starting a kernel."""
    target = target or FAST
    caches = list((SLIDES / ".jupyter_cache/executed").glob("*/base.ipynb"))
    if not caches:
        raise RuntimeError(
            "Nessuna esecuzione salvata: avvia prima la preview normale."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    # Keep fast rendering independent of normal renders and publishing.
    resources = [SLIDES / name for name in RESOURCES]
    resources += list(SLIDES.glob("_capitoli/*.qmd"))
    resources += [path for path in (SLIDES / "assets").rglob("*") if path.is_file()]
    for source in resources:
        destination = target.parent / source.relative_to(SLIDES)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists() or source.read_bytes() != destination.read_bytes():
            shutil.copy2(source, destination)
    # A corso.qmd beside corso.ipynb makes Quarto exclude the notebook from the
    # project. Keep the include manifest under an ignored, different basename.
    manifest = target.parent / "_source-links.qmd"
    if not manifest.exists() or manifest.read_bytes() != MAIN.read_bytes():
        shutil.copy2(MAIN, manifest)
    with tempfile.TemporaryDirectory(prefix="course-preview-") as scratch:
        source = Path(scratch) / "slides.qmd"
        converted = Path(scratch) / "slides.ipynb"
        source.write_text(expand_includes(MAIN))
        subprocess.run(
            ["quarto", "convert", str(source), "--output", str(converted)],
            cwd=SLIDES,
            check=True,
        )
        notebook = json.loads(converted.read_text())
    # Prefer the run matching the most cells; use the newest run to break ties.
    best, missing, best_rank = None, 0, None
    for path in caches:
        candidate, unmatched = reuse_outputs(notebook, json.loads(path.read_text()))
        rank = (-unmatched, path.stat().st_mtime)
        if best_rank is None or rank > best_rank:
            best, missing, best_rank = candidate, unmatched, rank
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(best, ensure_ascii=False))
    temporary.replace(target)
    print(
        "FAST: testo attuale e risultati salvati; nessun calcolo viene eseguito.",
        flush=True,
    )
    if missing:
        print(
            f"Attenzione: celle nuove/modificate senza output aggiornato: {missing}. "
            "Per ricalcolare, avvia senza --fast.",
            flush=True,
        )


def run_preview(target: Path, extra: list[str], fast: bool = False) -> None:
    """Watch the source slides while Quarto serves this preview's own project."""
    if fast:
        build_fast_preview(target)
        extra = [*extra, "--no-execute"]
    # Extra arguments go to quarto, e.g. --no-execute or --port 4200
    preview = subprocess.Popen(
        ["quarto", "preview", target.name, *extra], cwd=target.parent
    )
    seen = chapter_times()
    try:
        while preview.poll() is None:
            time.sleep(0.5)
            now = chapter_times()
            if now != seen:
                if fast:
                    build_fast_preview(target)
                else:
                    MAIN.touch()  # quarto sees corso.qmd change and renders again
                    now[MAIN] = MAIN.stat().st_mtime
                seen = now
    except KeyboardInterrupt:
        pass
    finally:
        if preview.poll() is None:
            preview.terminate()
        preview.wait()
    if preview.returncode:
        sys.exit(preview.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast", action="store_true", help="Reuse saved outputs without execution"
    )
    options, extra = parser.parse_known_args()
    if options.fast:
        FAST.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent previews must never move or clean each other's Reveal resources.
        with tempfile.TemporaryDirectory(prefix="session-", dir=FAST.parent) as session:
            run_preview(Path(session) / "corso.ipynb", extra, fast=True)
    else:
        run_preview(MAIN, extra)


if __name__ == "__main__":
    main()
