"""Read FAQ and documents from the PR FESR Emilia-Romagna website (Plone 6 REST API)."""

import re
from dataclasses import dataclass

import httpx

BASE_URL = "https://fesr.regione.emilia-romagna.it"
API_URL = f"{BASE_URL}/++api++"

# Funding calls live at /opportunita/opportunita-di-finanziamento/<year>/<call>/...
CALL_PATTERN = re.compile(r"/opportunita-di-finanziamento/\d{4}/([^/]+)")


@dataclass(frozen=True)
class Faq:
    question: str
    answer: str
    url: str
    call: str | None


def call_slug(url: str) -> str | None:
    """Return the funding call a URL belongs to, or None."""
    match = CALL_PATTERN.search(url)
    return match.group(1) if match else None


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


def search(client: httpx.Client, **query) -> list[dict]:
    """Search the site, following pagination until the last page."""
    url, params = f"{API_URL}/@search", {"b_size": 500, **query}
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
