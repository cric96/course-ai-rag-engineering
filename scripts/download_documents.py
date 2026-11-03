"""Download corpus PDFs and images into data/raw/fesr/{pdf,images}/<call>/.

Run scripts/download_faq.py first: calls with FAQ define the text corpus.
"""

import json
from dataclasses import asdict
from pathlib import Path

import httpx

from ragcourse.fesr import (
    BASE_URL,
    Document,
    call_url,
    download,
    list_documents,
    list_images,
    local_path,
)

FAQ_FILE = Path("data/raw/fesr/faq.jsonl")
PDF_DIR = Path("data/raw/fesr/pdf")
IMAGE_DIR = Path("data/raw/fesr/images")
INDEX_FILE = Path("data/raw/fesr/documents.jsonl")

CALLS_URL = f"{BASE_URL}/opportunita/opportunita-di-finanziamento"

# 2026 calls without FAQ folders, kept for their webinar slides (image-only pages).
VISUAL_CALLS = [
    f"{CALLS_URL}/2026/innovazione-nella-gestione-delle-sale-deputate-allo-svolgimento-di-rappresentazioni-cinematografiche-teatrali-musicali-e-artistiche",
    f"{CALLS_URL}/2026/recupero-e-incremento-del-patrimonio-di-edilizia-residenziale-pubblica-erp-e-di-edilizia-residenziale-sociale-ers",
    f"{CALLS_URL}/2026/sostegno-alla-produzione-di-opere-cinematografiche-e-audiovisive-anno-2026",
    f"{CALLS_URL}/2026/supporto-allutilizzo-di-fonti-rinnovabili-nelle-imprese",
]

# Programme texts: full PR FESR, its summary, strategic assessment.
PROGRAMME = f"{BASE_URL}/programmi-e-strategie/2021-2027/documenti/programma"

# Infographics with programme results: numbers that exist only as images.
IMAGE_FOLDERS = [
    f"{BASE_URL}/programmi-e-strategie/por-fesr/documenti/infografiche-assi",
    f"{BASE_URL}/programmi-e-strategie/2021-2027/immagini-pr-fesr-2021-2027",
]


def save_all(client: httpx.Client, documents: list[Document], root: Path) -> list:
    """Download documents under `root`; return their index rows."""
    rows = []
    for document in documents:
        path = local_path(document, root)
        download(client, document, path)
        rows.append({**asdict(document), "path": str(path)})
    return rows


def main() -> None:
    faq_calls = {call_url(json.loads(line)["url"]) for line in FAQ_FILE.open()}
    folders = sorted(faq_calls) + VISUAL_CALLS + [PROGRAMME]

    index = []
    with httpx.Client(timeout=120) as client:
        for folder in folders:
            documents = list_documents(client, folder)
            index += save_all(client, documents, PDF_DIR)
            print(f"{len(documents):>3} PDF    {folder.rsplit('/', 1)[1][:70]}")

        for folder in IMAGE_FOLDERS:
            # The 2021 archive repeats the same charts with older numbers: skip it.
            images = [d for d in list_images(client, folder) if "archivio" not in d.url]
            index += save_all(client, images, IMAGE_DIR)
            print(f"{len(images):>3} images {folder.rsplit('/', 1)[1][:70]}")

    # The index maps every local file back to its title and source URL.
    with INDEX_FILE.open("w", encoding="utf-8") as file:
        for row in index:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {len(index)} files, index in {INDEX_FILE}")


if __name__ == "__main__":
    main()
