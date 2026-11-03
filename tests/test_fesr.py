from pathlib import Path

import httpx

from ragcourse.fesr import (
    API_URL,
    BASE_URL,
    Document,
    Faq,
    call_slug,
    call_url,
    download,
    fetch_faqs,
    is_corpus_pdf,
    list_documents,
    local_path,
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


def test_call_url_strips_everything_below_the_call():
    assert call_url(f"{CALL_URL}/bando-e-modulistica/bando.pdf") == CALL_URL


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


def file_item(
    title: str, mime: str = "application/pdf", path: str = "bando.pdf"
) -> dict:
    # A "File" entry as returned by @search.
    return {"@id": f"{CALL_URL}/{path}", "title": title, "mime_type": mime}


def test_corpus_keeps_pdfs():
    assert is_corpus_pdf(file_item("Bando"))


def test_corpus_skips_office_files():
    assert not is_corpus_pdf(file_item("Modulo", mime="application/msword"))


def test_corpus_skips_faq_files_to_avoid_leaking_answers():
    assert not is_corpus_pdf(file_item("Faq bando.pdf"))
    assert not is_corpus_pdf(file_item("Risposte", path="faq-domande/risposte.pdf"))


def test_list_documents_searches_inside_the_call():
    items = [file_item("Bando"), file_item("Modulo", mime="application/msword")]
    routes = {f"/++api++{CALL_PATH}/@search": {"items": items}}
    client = fake_client(lambda request: routes[request.url.path])

    assert list_documents(client, CALL_URL) == [
        Document(title="Bando", url=f"{CALL_URL}/bando.pdf", call="bando-preciseu")
    ]


def test_local_path_groups_files_by_call_and_keeps_pdf_suffix():
    document = Document(title="Bando", url=f"{CALL_URL}/bando", call="bando-preciseu")
    assert local_path(document, Path("data")) == Path("data/bando-preciseu/bando.pdf")


def test_download_saves_file_once(tmp_path: Path):
    document = Document(title="Bando", url=f"{CALL_URL}/bando", call="bando-preciseu")
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return httpx.Response(200, content=b"%PDF-1.7")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    path = tmp_path / "bando.pdf"
    download(client, document, path)
    download(client, document, path)  # second call: already on disk

    assert path.read_bytes() == b"%PDF-1.7"
    assert requests == [f"{CALL_PATH}/bando/@@download/file"]
