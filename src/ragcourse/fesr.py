"""Read FAQ and documents from the PR FESR Emilia-Romagna website (Plone 6 REST API)."""

import re
from dataclasses import dataclass
from pathlib import Path

import httpx

BASE_URL = "https://fesr.regione.emilia-romagna.it"
API_URL = f"{BASE_URL}/++api++"

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

# Funding calls live at /opportunita/opportunita-di-finanziamento/<year>/<call>/...
CALL_PATTERN = re.compile(r".*/opportunita-di-finanziamento/\d{4}/[^/]+")


@dataclass(frozen=True)
class Faq:
    question: str
    answer: str
    url: str
    call: str | None


@dataclass(frozen=True)
class Document:
    title: str
    url: str
    call: str | None
    kind: str = "file"  # Plone field with the content: "file" (PDF) or "image"


def call_url(url: str) -> str | None:
    """Return the URL of the funding call a page belongs to, or None."""
    match = CALL_PATTERN.match(url)
    return match.group(0) if match else None


def call_slug(url: str) -> str | None:
    """Return the short name of the funding call a URL belongs to, or None."""
    call = call_url(url)
    return call.rsplit("/", 1)[1] if call else None


def parse_faq_structure(node: dict) -> list[Faq]:
    """Extract question/answer pairs from an `@faq-structure` API response."""
    if node.get("@type") == "Faq":
        answer = _answer_text(node)
        if not answer:
            return []
        url = node["@id"]
        return [Faq(node["title"].strip(), answer, url, call_slug(url))]

    # Folders (and the root) just contain more items: recurse.
    return [
        faq for child in node.get("items", []) for faq in parse_faq_structure(child)
    ]


def _answer_text(node: dict) -> str:
    # The answer is the text of the "slate" blocks, in page order.
    blocks = node.get("blocks", {})
    order = node.get("blocks_layout", {}).get("items", list(blocks))
    texts = [
        blocks[block_id].get("plaintext", "").strip()
        for block_id in order
        if blocks.get(block_id, {}).get("@type") == "slate"
    ]
    return "\n".join(text for text in texts if text)


def api_url(url: str) -> str:
    """Turn a public page URL into its REST API URL."""
    return url.replace(BASE_URL, API_URL, 1)


def search(client: httpx.Client, within: str = BASE_URL, **query) -> list[dict]:
    """Search the site (or only below `within`), following pagination."""
    url, params = f"{api_url(within)}/@search", {"b_size": 500, **query}
    items = []
    while url:
        page = client.get(url, params=params).raise_for_status().json()
        items += page["items"]
        # The "next" link already carries the whole query.
        url, params = page.get("batching", {}).get("next"), None
    return items


def fetch_faqs(client: httpx.Client) -> list[Faq]:
    """Download the FAQ of every funding call."""
    faqs = []
    for folder in search(client, portal_type="FaqFolder"):
        if call_slug(folder["@id"]) is None:
            continue  # FAQ not tied to a call (e.g. STAMI): skip
        response = client.get(api_url(folder["@id"]) + "/@faq-structure")
        faqs += parse_faq_structure(response.raise_for_status().json())
    return faqs


def is_corpus_pdf(item: dict) -> bool:
    """Keep PDFs only, never FAQ files: they contain the evaluation answers."""
    is_pdf = item.get("mime_type") == "application/pdf"
    is_faq = "faq" in (item["title"] + item["@id"]).lower()
    return is_pdf and not is_faq


def list_documents(client: httpx.Client, within: str) -> list[Document]:
    """List the corpus PDFs published below a page (e.g. a funding call)."""
    items = search(client, within, portal_type="File", metadata_fields="mime_type")
    return [
        Document(item["title"].strip(), item["@id"], call_slug(item["@id"]))
        for item in items
        if is_corpus_pdf(item)
    ]


def is_corpus_image(item: dict) -> bool:
    """Keep content images (infographics, charts), not logos."""
    url = item["@id"].lower()
    return url.endswith(IMAGE_SUFFIXES) and "logo" not in url and "loghi" not in url


def list_images(client: httpx.Client, within: str) -> list[Document]:
    """List the corpus images published below a page."""
    items = search(client, within, portal_type="Image")
    return [
        Document(item["title"].strip(), item["@id"], call_slug(item["@id"]), "image")
        for item in items
        if is_corpus_image(item)
    ]


def local_path(document: Document, root: Path) -> Path:
    """Where a document is saved: <root>/<call>/<file name>."""
    name = document.url.rstrip("/").rsplit("/", 1)[1]
    if document.kind == "file" and not name.endswith(".pdf"):
        name += ".pdf"
    # Documents outside calls (e.g. the programme) share one folder.
    return root / (document.call or "programma") / name


def download(client: httpx.Client, document: Document, path: Path) -> None:
    """Save a document to `path`, unless it is already there."""
    if path.exists():
        return
    url = f"{document.url}/@@download/{document.kind}"
    response = client.get(url, follow_redirects=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.raise_for_status().content)
