"""Live preview of the course slides, also when a chapter changes.

quarto preview watches corso.qmd but not the chapters it includes:
when a chapter is saved, we touch corso.qmd so the preview renders again.

    uv run python scripts/preview_slides.py                # real outputs (cached)
    uv run python scripts/preview_slides.py --no-execute   # text only, no outputs
"""

import subprocess
import sys
import time
from pathlib import Path

SLIDES = Path(__file__).resolve().parent.parent / "slides"
MAIN = SLIDES / "corso.qmd"


def chapter_times() -> dict[Path, float]:
    """Last modification time of each chapter."""
    return {path: path.stat().st_mtime for path in SLIDES.glob("_capitoli/*.qmd")}


def main() -> None:
    # Extra arguments go to quarto, e.g. --no-execute or --port 4200
    preview = subprocess.Popen(
        ["quarto", "preview", MAIN.name, *sys.argv[1:]], cwd=SLIDES
    )
    seen = chapter_times()
    try:
        while preview.poll() is None:
            time.sleep(0.5)
            now = chapter_times()
            if now != seen:
                MAIN.touch()  # quarto sees corso.qmd change and renders again
                seen = now
    except KeyboardInterrupt:
        pass
    finally:
        preview.terminate()
        preview.wait()


if __name__ == "__main__":
    main()
