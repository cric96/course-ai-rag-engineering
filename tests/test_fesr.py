import httpx

from ragcourse.fesr import (
    API_URL,
    BASE_URL,
    Faq,
    call_slug,
    fetch_faqs,
    parse_faq_structure,
    search,
)

CALL_PATH = "/opportunita/opportunita-di-finanziamento/2026/bando-preciseu"
CALL_URL = BASE_URL + CALL_PATH


def slate(text: str) -> dict:
    return {"@type": "slate", "plaintext": text}


def faq(title: str, blocks: dict, layout: list[str] | None = None) -> dict:
    # Mirrors a Plone "Faq" item: question in title, answer in blocks.
    return {
        "@type": "Faq",
        "@id": f"{CALL_URL}/faq-domande-e-risposte/q1",
        "title": title,
        "blocks": blocks,
        "blocks_layout": {"items": layout or list(blocks)},
    }


def folder(*items: dict) -> dict:
    return {"@type": "FaqFolder", "items": list(items)}


def test_call_slug_extracts_call_from_url():
    assert call_slug(f"{CALL_URL}/faq/q1") == "bando-preciseu"


def test_call_slug_is_none_outside_funding_calls():
    url = "https://fesr.regione.emilia-romagna.it/progetti-attivita/stami/faq"
    assert call_slug(url) is None


def test_parse_returns_question_answer_and_call():
    payload = {"items": [folder(faq("Chi può partecipare?", {"a": slate("Le PMI.")}))]}

    assert parse_faq_structure(payload) == [
        Faq(
            question="Chi può partecipare?",
            answer="Le PMI.",
            url=f"{CALL_URL}/faq-domande-e-risposte/q1",
            call="bando-preciseu",
        )
    ]


def test_parse_joins_slate_blocks_in_layout_order():
    blocks = {"t": {"@type": "title"}, "b": slate("Secondo."), "a": slate("Primo.")}
    payload = {"items": [folder(faq("Domanda?", blocks, layout=["t", "a", "b"]))]}

    [item] = parse_faq_structure(payload)
    assert item.answer == "Primo.\nSecondo."


def test_parse_skips_faq_without_answer():
    payload = {"items": [folder(faq("Domanda?", {"a": slate("  ")}))]}
    assert parse_faq_structure(payload) == []


def test_parse_walks_nested_folders():
    inner = folder(faq("Domanda?", {"a": slate("Risposta.")}))
    payload = {"items": [folder(inner)]}
    assert len(parse_faq_structure(payload)) == 1


def fake_client(respond) -> httpx.Client:
    # Offline client: `respond(request)` returns the JSON the "site" sends back.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=respond(request))

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_follows_pagination():
    pages = {
        None: {
            "items": [{"@id": "a"}],
            "batching": {"next": f"{API_URL}/@search?b_start=1"},
        },
        "1": {"items": [{"@id": "b"}]},
    }
    client = fake_client(lambda request: pages[request.url.params.get("b_start")])

    assert [item["@id"] for item in search(client, portal_type="File")] == ["a", "b"]


def test_fetch_faqs_reads_only_folders_of_funding_calls():
    call_folder = f"{CALL_URL}/faq-domande-e-risposte"
    other_folder = f"{BASE_URL}/progetti-attivita/stami/faq"
    structure = {"items": [folder(faq("Domanda?", {"a": slate("Risposta.")}))]}
    routes = {
        "/++api++/@search": {"items": [{"@id": call_folder}, {"@id": other_folder}]},
        f"/++api++{CALL_PATH}/faq-domande-e-risposte/@faq-structure": structure,
    }
    client = fake_client(lambda request: routes[request.url.path])

    assert [item.call for item in fetch_faqs(client)] == ["bando-preciseu"]
