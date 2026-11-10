"""How a past turn is shown: answer, judgment, cost and what happened behind it."""

import streamlit as st

from chat_ui.pipeline import Settings, Turn
from chat_ui.resources import ANSWER_MODELS, pages
from ragcourse.chat import cited, dollars, link_citations
from ragcourse.costs import PRICING_CHECKED, PRICING_URL
from ragcourse.evaluation import gold_found, page_id
from ragcourse.generation import ABSTAIN
from ragcourse.models import JUDGE_MODEL
from ragcourse.ocr import page_picture

EVIDENCE = {"text": "testo", "table": "tabella", "image": "immagine", "none": "assente"}


def show_turn(turn: Turn) -> None:
    """A past turn, exactly as it looked when it was answered."""
    with st.chat_message("user"):
        st.markdown(turn.question)
    with st.chat_message("assistant"):
        with st.status(summary(turn.settings), state="complete", expanded=False):
            for text in turn.steps:
                st.markdown(text)
        st.markdown(link_citations(turn.answer, turn.chunks))
        show_judgment(turn)
        show_cost(turn)
        show_details(turn)


def summary(settings: Settings) -> str:
    """The levers used for this answer, in one line."""
    levers = ["senza documenti"]
    if settings.rag:
        levers = [
            settings.mode,
            "con OCR" if settings.ocr else "senza OCR",
            f"reranking: {settings.reranker}",
            f"{settings.k} chunk",
            f"prompt {settings.prompt}",
        ]
    if settings.thinking != "predefinito":
        levers.append(f"ragionamento {settings.thinking}")
    if settings.judge:
        levers.append("con giudice")
    return "Dietro le quinte · " + ", ".join([ANSWER_MODELS[settings.model], *levers])


def show_judgment(turn: Turn) -> None:
    """Keep a model's assessment distinct from an independently verified fact."""
    if turn.judge_error:
        st.warning("Il giudice non è riuscito a valutare questa risposta.")
    elif turn.judgment:
        st.markdown(f"**⚖️ Giudizio AI: {turn.judgment.verdict}**")
        st.caption(f"{ANSWER_MODELS[JUDGE_MODEL]} · {turn.judge_basis}")
        st.markdown(link_citations(turn.judgment.reason, turn.chunks))


def show_cost(turn: Turn) -> None:
    """Show each API stage, without treating missing usage as zero."""
    costs = turn.api_costs
    if not costs:
        return
    complete = all(c.usd is not None for c in costs)
    total = sum(c.usd for c in costs) if complete else None
    st.caption(f"💰 Costo API stimato: **{dollars(total)}**")
    if len(costs) > 1:
        st.caption(" · ".join(f"{c.stage}: {dollars(c.usd)}" for c in costs))
    if not complete:
        st.caption("Consumi dei token non disponibili per una o più chiamate.")
    st.caption(
        f"[Listino Google standard]({PRICING_URL}), verificato il {PRICING_CHECKED}. "
        "L'eventuale piano gratuito può azzerare il costo."
    )


def show_details(turn: Turn) -> None:
    """One tab per view of the answer: sources, chunks, prompt, FAQ."""
    tabs = {}
    if turn.chunks:
        tabs["Fonti citate"] = show_sources
        tabs[f"Chunk letti ({len(turn.chunks)})"] = show_chunks
    tabs["Il prompt"] = show_prompt
    if turn.faq:
        tabs["La FAQ"] = show_faq
    for tab, show_tab in zip(st.tabs(list(tabs)), tabs.values(), strict=True):
        with tab:
            show_tab(turn)
    if turn.answer.startswith(ABSTAIN):
        st.caption(
            "Il modello si è astenuto: nei chunk letti non ha trovato la risposta."
        )


def show_sources(turn: Turn) -> None:
    """Thumbnails of the cited pages, each linked to the PDF at that page."""
    by_page = {}  # several cited chunks may come from the same page
    for n in cited(turn.answer, turn.chunks):
        meta = turn.chunks[n - 1].metadata
        by_page.setdefault((meta["url"], meta["page"]), []).append(n)
    if not by_page:
        st.caption("La risposta non cita nessun chunk.")
    cited_pages = list(by_page.items())
    for row in range(0, len(cited_pages), 4):
        columns = st.columns(4)  # the last row may have fewer pages than columns
        for column, (key, refs) in zip(
            columns, cited_pages[row : row + 4], strict=False
        ):
            page, link = pages()[key], page_id(*key)  # the link opens the PDF there
            with column:
                st.image(page_picture(page, dpi=40), link=link, width="stretch")
                st.markdown(
                    f"**{' '.join(f'[{n}]' for n in sorted(refs))}** "
                    f"[{page['title']}, p. {key[1]}]({link})"
                )


def show_chunks(turn: Turn) -> None:
    """What the model read: position before/after reranking, cited or not, OCR."""
    numbers = cited(turn.answer, turn.chunks)
    rows = [
        {
            "n": n,
            "citato": n in numbers,
            "documento": c.metadata["title"],
            "pagina": c.metadata["page"],
            "posizione nella ricerca": position + 1,
            "letto con OCR": not pages()[(c.metadata["url"], c.metadata["page"])][
                "text"
            ],
            "testo": c.page_content,
        }
        for n, (c, position) in enumerate(
            zip(turn.chunks, turn.positions, strict=True), start=1
        )
    ]
    st.dataframe(
        rows,
        hide_index=True,
        column_config={"testo": st.column_config.TextColumn(width="large")},
    )
    st.caption(
        "Il testo completo dei chunk, come lo legge il modello: scheda «Il prompt»."
    )


def show_prompt(turn: Turn) -> None:
    """Exactly what the model received: the rules, the numbered chunks, the question."""
    for message in turn.prompt:
        st.caption({"system": "Istruzioni", "human": "Messaggio"}[message.type])
        st.code(message.text, language=None, wrap_lines=True, height=420)


def show_faq(turn: Turn) -> None:
    """The official answer, and whether its page reached the model."""
    faq = turn.faq
    st.markdown(f"**Risposta ufficiale** (FAQ, prova: {EVIDENCE[faq['evidence']]})")
    st.markdown(faq["answer"] or "_La FAQ rimanda ad altro: nei documenti non c'è._")
    if faq["gold"] and turn.chunks:
        found = gold_found(turn.chunks, faq)
        st.markdown(
            ("✅ La pagina giusta è" if found else "❌ La pagina giusta non è")
            + " tra i chunk letti dal modello."
        )
